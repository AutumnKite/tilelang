#pragma once

#include <tvm/runtime/logging.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "../../op/builtin.h"
#include "../../op/utils.h"
#include "./ir_structure.h"
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;
using ffi::Array;
using ffi::Map;

// Calculate the size in bytes of a buffer. Returns 0 if the size cannot be
// determined at compile time.
inline size_t GetBufferSize(const Buffer &buffer) {
  size_t elem_size = buffer->dtype.bits() * buffer->dtype.lanes() / 8;
  PrimExpr size = IntImm(DataType::Int(64), elem_size);
  for (const auto &dim : buffer->shape) {
    size *= dim;
  }
  arith::Analyzer analyzer;
  auto size_val = analyzer.Simplify(size);
  if (const auto *int_imm = size_val.as<IntImmNode>()) {
    return int_imm->value;
  } else {
    return 0;
  }
}

// Helper function to rewrite alloc_buffer for multi-version support
inline Buffer RewriteAllocBuffer(const Buffer &buffer, int num_versions) {
  // Create a copy of the buffer
  ObjectPtr<BufferNode> new_buffer =
      tvm::ffi::make_object<BufferNode>(*(buffer.get()));

  // Add num_versions as first dimension
  new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_versions));

  // Update strides if they exist
  if (!new_buffer->strides.empty()) {
    ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
    PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
    new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
  }

  return Buffer(new_buffer);
}

// Create a barrier buffer allocation and return the Buffer object.
// shape = (num_versions,) so that each pipeline stage has its own barrier slot,
// accessed via BufferLoad(buffer, {stage_index}).
static Buffer makeBarrierBuffer(PrimExpr arrive_count, const std::string &name,
                                int num_versions,
                                std::vector<Buffer> &barrier_buffers,
                                Map<ObjectRef, ObjectRef> &barrier_map) {
  Array<PrimExpr> shape = {num_versions};
  DataType dtype = DataType::UInt(64);
  Type ptr_type = PointerType(PrimType(dtype), "shared.barrier");
  Var handle(name, ptr_type);
  Array<ObjectRef> arrive_counts;
  for (int i = 0; i < num_versions; ++i) {
    arrive_counts.push_back(arrive_count);
  }
  barrier_map.Set(handle, arrive_counts);
  Buffer buffer =
      Buffer(handle, dtype, shape, {}, PrimExpr(), name, 0, 0, kDefault);
  barrier_buffers.push_back(buffer);
  return buffer;
}

inline bool IsEvaluateZero(const tvm::tir::Stmt &stmt) {
  if (const EvaluateNode *eval_node = stmt.as<EvaluateNode>()) {
    if (is_const_int(eval_node->value, 0)) {
      return true;
    }
  }
  return false;
}

// Structure to store loop nesting information
struct LoopNestingInfo {
  std::vector<Var> loop_vars;
  std::vector<PrimExpr> loop_starts;
  std::vector<PrimExpr> loop_steps;
  std::vector<PrimExpr> loop_extents;

  // Add a loop to the nesting info
  void AddLoop(const ForNode *for_node) {
    loop_vars.push_back(for_node->loop_var);
    loop_starts.push_back(for_node->min);
    loop_steps.push_back(for_node->step.has_value()
                             ? for_node->step.value()
                             : IntImm(DataType::Int(32), 1));
    loop_extents.push_back(for_node->extent);
  }

  // Remove the innermost loop
  void PopLoop() {
    if (!loop_vars.empty()) {
      loop_vars.pop_back();
      loop_starts.pop_back();
      loop_steps.pop_back();
      loop_extents.pop_back();
    }
  }

  PrimExpr CalculateIterationCount() const {
    PrimExpr total_iter = IntImm(DataType::Int(32), 0);
    PrimExpr total_multiplier = IntImm(DataType::Int(32), 1);
    for (size_t i = loop_vars.size(); i-- > 0;) {
      PrimExpr normalized_iter =
          indexdiv(loop_vars[i] - loop_starts[i], loop_steps[i]);
      if (i == static_cast<int>(loop_vars.size()) - 1) {
        total_iter = normalized_iter;
      } else {
        total_iter = total_iter + normalized_iter * total_multiplier;
      }
      total_multiplier = total_multiplier * loop_extents[i];
    }
    return total_iter;
  }
};

// Structure to store multi-version buffer information
struct MultiVersionBufferInfo {
  Buffer buffer;
  int num_versions;
  Buffer new_buffer;

  MultiVersionBufferInfo(Buffer buffer, int num_versions, Buffer new_buffer)
      : buffer(buffer), num_versions(num_versions), new_buffer(new_buffer) {}
};

// Barrier dependency analysis function declarations
static void AnalyzeAndInsertBarriers(
    IRStructure *node, int &next_barrier_id,
    std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map,
    const std::vector<PrimExpr> &thread_count, LoopNestingInfo &loop_info,
    std::vector<MultiVersionBufferInfo> &buffer_infos,
    Buffer neutral_sync_shared_barrier, bool is_root = false);
static void AnalyzeSequenceNodeBarriers(
    SequenceNode *seq, int &next_barrier_id,
    std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map,
    const std::vector<PrimExpr> &thread_count, LoopNestingInfo &loop_info,
    std::vector<MultiVersionBufferInfo> &buffer_infos,
    Buffer neutral_sync_shared_barrier, bool is_root = false);
static void AnalyzeControlNodeBarriers(
    ControlNode *ctrl, int &next_barrier_id,
    std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map,
    const std::vector<PrimExpr> &thread_count, LoopNestingInfo &loop_info,
    std::vector<MultiVersionBufferInfo> &buffer_infos,
    Buffer neutral_sync_shared_barrier, bool is_root = false);

// Create a barrier_arrive statement for the given barrier expression
// Equivalent to T.barrier_arrive(barrier_expr) in Python
// barrier_expr should be BufferLoad(barrier_buffer, {0}) where barrier_buffer
// is allocated with makeAllocBarrier
static Stmt makeBarrierArrive(PrimExpr barrier_expr, int cta_id = -1,
                              const PrimExpr &pred = 1) {
  Array<PrimExpr> args = {std::move(barrier_expr)};
  if (cta_id != -1) {
    args.push_back(cta_id);
    args.push_back(pred);
  }
  return Evaluate(
      Call(DataType::Handle(), builtin::ptx_arrive_barrier(), args));
}

static Stmt makeTcgen05MmaArrive(Buffer barrier_buffer,
                                 PrimExpr offset = IntImm(DataType::Int(32),
                                                          0)) {
  auto access_ptr = barrier_buffer.access_ptr(3, DataType::Handle(), 1, offset);
  return Evaluate(Call(DataType::Handle(), tcgen05_mma_arrive(), {access_ptr}));
}

// Create a barrier_wait statement for the given barrier expression and parity
// Equivalent to T.barrier_wait(barrier_expr, parity) in Python
// barrier_expr should be BufferLoad(barrier_buffer, {0}) where barrier_buffer
// is allocated with makeAllocBarrier
static Stmt makeBarrierWait(PrimExpr barrier_expr, PrimExpr parity) {
  auto call = Call(DataType::Handle(), mbarrier_wait_parity(),
                   {std::move(barrier_expr), std::move(parity)});
  return Evaluate(call);
}

static bool IsRegularSharedScope(const String &scope) {
  return scope == "shared" || scope == "shared.dyn";
}

static bool IsTmemScope(const String &scope) { return scope == "shared.tmem"; }

static bool HasWriteReadDependencyByScope(
    const IRStructure *producer, const IRStructure *consumer,
    const std::function<bool(const String &)> &scope_matcher) {
  if (!producer || !consumer) {
    return false;
  }

  auto producer_writes = producer->GetWriteRegions();
  auto consumer_reads = consumer->GetReadRegions();

  for (const auto &write_region : producer_writes) {
    const Buffer &write_buffer = write_region->buffer;
    if (!scope_matcher(write_buffer.scope())) {
      continue;
    }
    for (const auto &read_region : consumer_reads) {
      if (write_buffer.same_as(read_region->buffer)) {
        return true;
      }
    }
  }
  return false;
}

static bool HasSharedWriteReadDependency(const IRStructure *producer,
                                         const IRStructure *consumer) {
  return HasWriteReadDependencyByScope(
      producer, consumer,
      [](const String &scope) { return IsRegularSharedScope(scope); });
}

static bool HasTmemWriteReadDependency(const IRStructure *producer,
                                       const IRStructure *consumer) {
  return HasWriteReadDependencyByScope(
      producer, consumer,
      [](const String &scope) { return IsTmemScope(scope); });
}

static Stmt InsertBarriersForNeutralSyncWithDependency(
    Stmt producer_body, Stmt consumer_body,
    std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, PrimExpr total_thread_count,
    bool need_regular_barrier, bool need_tmem_barrier,
    Buffer neutral_sync_shared_barrier = Buffer(), Var thread_var = Var(),
    PrimExpr tensor_core_wg_start = PrimExpr(),
    PrimExpr tensor_core_wg_end = PrimExpr()) {
  if (!need_regular_barrier && !need_tmem_barrier) {
    return SeqStmt({producer_body, consumer_body});
  }

  std::vector<Stmt> arrive_stmts;
  std::vector<Stmt> wait_stmts;

  if (need_regular_barrier) {
    Buffer barrier_buffer =
        neutral_sync_shared_barrier.defined()
            ? neutral_sync_shared_barrier
            : makeBarrierBuffer(total_thread_count,
                                "neutral_sync_shared_barrier", 1,
                                barrier_buffers, barrier_map);
    PrimExpr barrier_load = BufferLoad(barrier_buffer, {0});
    arrive_stmts.push_back(makeBarrierArrive(barrier_load));
    wait_stmts.push_back(makeBarrierWait(barrier_load, 0));
  }

  if (need_tmem_barrier) {
    // TMEM barrier: arrive_count = 1 (only tensor core warp arrives)
    Buffer barrier_buffer = makeBarrierBuffer(IntImm(DataType::Int(32), 1),
                                              "neutral_sync_tmem_barrier", 1,
                                              barrier_buffers, barrier_map);
    PrimExpr barrier_load = BufferLoad(barrier_buffer, {0});

    // tcgen05_mma_arrive only in tensor core warpgroup
    Stmt tmem_arrive = makeTcgen05MmaArrive(barrier_buffer);
    if (tensor_core_wg_start.defined() && tensor_core_wg_end.defined()) {
      // Wrap in if condition for tensor core warpgroup only
      // Condition: tensor_core_wg_start <= thread_idx < tensor_core_wg_end
      tmem_arrive = IfThenElse((thread_var >= tensor_core_wg_start) &&
                                   (thread_var < tensor_core_wg_end),
                               tmem_arrive, Evaluate(0));
    }
    arrive_stmts.push_back(tmem_arrive);
    wait_stmts.push_back(makeBarrierWait(barrier_load, 0));
  }

  std::vector<Stmt> stmts;
  stmts.push_back(producer_body);
  stmts.insert(stmts.end(), arrive_stmts.begin(), arrive_stmts.end());
  stmts.insert(stmts.end(), wait_stmts.begin(), wait_stmts.end());
  stmts.push_back(consumer_body);
  return SeqStmt(stmts);
}

// Insert barriers between neutral tasks and warpgroup-specific work
// This ensures neutral tasks complete before any warpgroup-specific work begins
static Stmt InsertBarriersForNeutralSync(Stmt neutral_body, Stmt warpgroup_body,
                                         std::vector<Buffer> &barrier_buffers,
                                         Map<ObjectRef, ObjectRef> &barrier_map,
                                         PrimExpr total_thread_count,
                                         Buffer neutral_sync_shared_barrier) {
  return InsertBarriersForNeutralSyncWithDependency(
      neutral_body, warpgroup_body, barrier_buffers, barrier_map,
      total_thread_count, true, false, neutral_sync_shared_barrier);
}

// StmtExprMutator to rewrite BufferLoad/BufferStore for multi-version buffers
class MultiBufferAccessRewriter : public StmtExprMutator {
public:
  MultiBufferAccessRewriter(
      const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
          &multi_buffer,
      PrimExpr iteration)
      : multi_buffer_(multi_buffer), iteration_(iteration) {}

private:
  PrimExpr VisitExpr_(const BufferLoadNode *op) override {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));

    // Check if this buffer is in multi_buffer
    if (multi_buffer_.find(load->buffer) != multi_buffer_.end()) {
      // Add version_index as first dimension
      auto *n = load.CopyOnWrite();
      n->buffer = multi_buffer_.at(load->buffer);
      auto num_versions = n->buffer->shape[0];
      n->indices.insert(n->indices.begin(), indexmod(iteration_, num_versions));
    }

    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) override {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));

    // Check if this buffer is in multi_buffer
    if (multi_buffer_.find(store->buffer) != multi_buffer_.end()) {
      // Add version_index as first dimension
      auto *n = store.CopyOnWrite();
      n->buffer = multi_buffer_.at(store->buffer);
      auto num_versions = n->buffer->shape[0];
      n->indices.insert(n->indices.begin(), indexmod(iteration_, num_versions));
    }

    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) override {
    // Check if this is a tl.tileop.region call
    static const auto region_op = Op::Get("tl.tileop.region");
    if (op->op.same_as(region_op)) {
      // Handle tl.tileop.region call for multi-version buffers
      // args[0] = buffer (BufferLoad), args[1] = access_type (1: read, 2:
      // write, 3: read/write) args[2..] = extents
      if (op->args.size() >= 2) {
        // Check if the buffer in BufferLoad needs multi-version support
        if (const auto *buffer_load = op->args[0].as<BufferLoadNode>()) {
          auto it = multi_buffer_.find(buffer_load->buffer);
          if (it != multi_buffer_.end()) {
            // This buffer needs multi-version support
            // Create new arguments array
            Array<PrimExpr> new_args;

            // Add the updated BufferLoad (already processed by VisitExpr)
            new_args.push_back(VisitExpr(op->args[0]));

            // Add access_type (unchanged)
            new_args.push_back(VisitExpr(op->args[1]));

            // Add extent for version dimension (value = 1)
            new_args.push_back(IntImm(DataType::Int(32), 1));

            // Add existing extents (if any)
            for (size_t i = 2; i < op->args.size(); i++) {
              new_args.push_back(VisitExpr(op->args[i]));
            }

            // Create new Call node with updated arguments
            return Call(op->dtype, op->op, new_args, op->annotations, op->span);
          }
        }
      }
    }

    // For other Call nodes, use default processing
    return StmtExprMutator::VisitExpr_(op);
  }

  const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      &multi_buffer_;
  PrimExpr iteration_;
};

// Recursive function to rewrite BufferLoad/BufferStore in TaskNode stmts
static void RewriteTaskNodeBuffers(
    IRStructure *node,
    const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
        &multi_buffer,
    PrimExpr iteration) {
  if (!node)
    return;

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);

    // Apply MultiBufferAccessRewriter to all stmts in the task
    MultiBufferAccessRewriter rewriter(multi_buffer, iteration);
    for (auto &stmt : task->stmts) {
      stmt = rewriter(stmt);
    }
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);
    for (auto &child : seq->children) {
      RewriteTaskNodeBuffers(child.get(), multi_buffer, iteration);
    }
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node);
    RewriteTaskNodeBuffers(ctrl->child.get(), multi_buffer, iteration);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    RewriteTaskNodeBuffers(wrapper->child.get(), multi_buffer, iteration);
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    RewriteTaskNodeBuffers(unit->child.get(), multi_buffer, iteration);
  } else if (node->IsIf()) {
    auto if_node = static_cast<IfNode *>(node);
    if (if_node->then_child)
      RewriteTaskNodeBuffers(if_node->then_child.get(), multi_buffer,
                             iteration);
    if (if_node->else_child)
      RewriteTaskNodeBuffers(if_node->else_child.get(), multi_buffer,
                             iteration);
  }
}

// Rewrite the gemm call's mbar argument (arg[16]) in a TaskNode to use the
// allocated barrier expression from BarrierManager.
// This is used for TCGEN05MMA where the gemm needs to reference the correct
// mbarrier for synchronization.
static void RewriteGemmMbar(TaskNode *task, PrimExpr mbar_expr) {
  static const auto gemm_op = Op::Get("tl.tileop.gemm");
  static const auto wgmma_gemm_op = Op::Get("tl.tileop.wgmma_gemm");
  static const auto tcgen05_gemm_op = Op::Get("tl.tileop.tcgen05_gemm");

  class GemmMbarRewriter : public StmtExprMutator {
  public:
    GemmMbarRewriter(PrimExpr mbar_expr) : mbar_expr_(std::move(mbar_expr)) {}

  private:
    PrimExpr VisitExpr_(const CallNode *op) override {
      static const auto gemm_op = Op::Get("tl.tileop.gemm");
      static const auto wgmma_gemm_op = Op::Get("tl.tileop.wgmma_gemm");
      static const auto tcgen05_gemm_op = Op::Get("tl.tileop.tcgen05_gemm");

      if ((op->op.same_as(gemm_op) || op->op.same_as(wgmma_gemm_op) ||
           op->op.same_as(tcgen05_gemm_op)) &&
          op->args.size() > 16) {
        Array<PrimExpr> new_args;
        for (size_t i = 0; i < op->args.size(); ++i) {
          if (i == 16) {
            // Replace mbar argument with the barrier expression
            new_args.push_back(mbar_expr_);
          } else {
            new_args.push_back(VisitExpr(op->args[i]));
          }
        }
        return Call(op->dtype, op->op, new_args, op->annotations, op->span);
      }
      return StmtExprMutator::VisitExpr_(op);
    }

    PrimExpr mbar_expr_;
  };

  GemmMbarRewriter rewriter(mbar_expr);
  for (auto &stmt : task->stmts) {
    stmt = rewriter(stmt);
  }
}

static void RewriteCopyMbar(TaskNode *task, PrimExpr mbar_expr) {
  static const auto copy_op = Op::Get("tl.tileop.copy");
  static const auto tma_copy_op = Op::Get("tl.tileop.tma_copy");

  class CopyMbarRewriter : public StmtExprMutator {
  public:
    CopyMbarRewriter(PrimExpr mbar_expr) : mbar_expr_(std::move(mbar_expr)) {}

  private:
    PrimExpr VisitExpr_(const CallNode *op) override {
      if (op->op.same_as(copy_op)) {
        auto new_ann = op->annotations;
        new_ann.Set("barrier", mbar_expr_);
        return Call(op->dtype, tma_copy_op, op->args, new_ann, op->span);
      }
      return StmtExprMutator::VisitExpr_(op);
    }

    PrimExpr mbar_expr_;
  };

  CopyMbarRewriter rewriter(mbar_expr);
  for (auto &stmt : task->stmts) {
    stmt = rewriter(stmt);
  }
}

// Helper function to insert a statement into ScheduleUnit's stmts
static void InsertStatementIntoScheduleUnit(ScheduleUnit *task,
                                            const Stmt &stmt, bool at_beginning,
                                            int warpgroup_id) {
  if (at_beginning) {
    task->before[warpgroup_id].insert(task->before[warpgroup_id].begin(), stmt);
  } else {
    task->after[warpgroup_id].push_back(stmt);
  }
}

// Barrier dependency analysis implementation
static void
AnalyzeAndInsertBarriers(IRStructure *node, int &next_barrier_id,
                         std::vector<Buffer> &barrier_buffers,
                         Map<ObjectRef, ObjectRef> &barrier_map,
                         const std::vector<PrimExpr> &thread_count,
                         LoopNestingInfo &loop_info,
                         std::vector<MultiVersionBufferInfo> &buffer_infos,
                         Buffer neutral_sync_shared_barrier, bool is_root) {
  if (!node)
    return;

  if (node->IsSequence()) {
    AnalyzeSequenceNodeBarriers(static_cast<SequenceNode *>(node),
                                next_barrier_id, barrier_buffers, barrier_map,
                                thread_count, loop_info, buffer_infos,
                                neutral_sync_shared_barrier, is_root);
  } else if (node->IsControl()) {
    AnalyzeControlNodeBarriers(static_cast<ControlNode *>(node),
                               next_barrier_id, barrier_buffers, barrier_map,
                               thread_count, loop_info, buffer_infos,
                               neutral_sync_shared_barrier, is_root);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    AnalyzeAndInsertBarriers(
        wrapper->child.get(), next_barrier_id, barrier_buffers, barrier_map,
        thread_count, loop_info, buffer_infos, neutral_sync_shared_barrier);
  } else if (node->IsIf()) {
    auto if_node = static_cast<IfNode *>(node);
    if (if_node->then_child) {
      AnalyzeAndInsertBarriers(if_node->then_child.get(), next_barrier_id,
                               barrier_buffers, barrier_map, thread_count,
                               loop_info, buffer_infos,
                               neutral_sync_shared_barrier);
    }
    if (if_node->else_child) {
      AnalyzeAndInsertBarriers(if_node->else_child.get(), next_barrier_id,
                               barrier_buffers, barrier_map, thread_count,
                               loop_info, buffer_infos,
                               neutral_sync_shared_barrier);
    }
  } else if (node->IsTask()) {
    // For TaskNode, nothing to do at this level
  } else {
    LOG(FATAL);
  }
}

static TaskNode *GetInnerTask(ScheduleUnit *unit) {
  std::vector<TaskNodeWithContext> task_contexts;
  CollectAllTaskNodesWithContext(unit, task_contexts);
  if (task_contexts.size() == 1 && task_contexts[0].control_node == nullptr) {
    return task_contexts[0].task;
  } else {
    return nullptr;
  }
}

struct SyncInfo {
  int distance;             // the distance of iterations
  Buffer buffer;            // the buffer that requires synchronization
  const TaskNode *producer; // the innermost task that needs to be waited on
  const TaskNode *consumer; // the innermost task that needs to wait
  int buffer_versions; // the number of versions for the buffer (for calculating
                       // barrier slots)

  SyncInfo(int distance, Buffer buffer, const TaskNode *producer,
           const TaskNode *consumer, int buffer_versions)
      : distance(distance), buffer(buffer), producer(producer),
        consumer(consumer), buffer_versions(buffer_versions) {}

  // Define operator< for set
  bool operator<(const SyncInfo &other) const {
    if (distance != other.distance) {
      return distance < other.distance;
    }
    if (buffer->name != other.buffer->name) {
      return buffer->name < other.buffer->name;
    }
    if (producer != other.producer) {
      return producer < other.producer;
    }
    if (consumer != other.consumer) {
      return consumer < other.consumer;
    }
    return buffer_versions < other.buffer_versions;
  }
};

static auto
GetSyncInfos(const std::vector<ScheduleUnit *> &units, int num_wgs,
             const std::unordered_map<Buffer, int, ObjectPtrHash,
                                      ObjectPtrEqual> &buffer_num_versions = {},
             bool is_loop = false) {
  std::set<Buffer> buffers;
  for (auto *unit : units) {
    for (const auto &buffer_access :
         unit->GetBufferAccessInfo(num_wgs, SchedulePhase::kBody)) {
      buffers.insert(buffer_access.buffer);
    }
  }
  std::map<std::pair<ScheduleUnit *, int>,
           std::map<std::pair<ScheduleUnit *, int>, std::set<SyncInfo>>>
      sync_infos;
  for (const auto &buffer : buffers) {
    int num_versions = 1;
    auto it = buffer_num_versions.find(buffer);
    if (it != buffer_num_versions.end()) {
      num_versions = it->second;
    }
    std::vector<ScheduleUnit *> last_read_unit(num_wgs, nullptr);
    std::vector<std::set<const TaskNode *>> last_read_unit_tasks(num_wgs);
    ScheduleUnit *last_write_unit = nullptr;
    std::vector<std::pair<int, std::set<const TaskNode *>>>
        last_write_unit_wg_tasks;
    std::vector<bool> waited_write_wgs(num_wgs, false);
    for (int iter = 0; iter < (is_loop ? 2 : 1); ++iter) {
      for (ScheduleUnit *unit : units) {
        // Add dependencies for RAW and WAR between this unit and the last
        // writer/reader units
        for (int wg_id = 0; wg_id < num_wgs; ++wg_id) {
          int distance = iter ? num_versions : 0;
          // RAW: unit reads buffer, wait for last writer
          auto first_reads = unit->GetFirstAccessTasks(
              buffer, /*is_write=*/false, wg_id, SchedulePhase::kBody);
          if (!first_reads.empty() && last_write_unit != nullptr &&
              !waited_write_wgs[wg_id]) {
            for (auto *consumer : first_reads) {
              for (const auto &[last_write_wg_id, last_write_unit_tasks] :
                   last_write_unit_wg_tasks) {
                for (auto *producer : last_write_unit_tasks) {
                  if (IsRegisterBuffer(buffer) && wg_id != last_write_wg_id) {
                    // Skip cross-warpgroup dependency for register buffers
                    continue;
                  }
                  sync_infos[{last_write_unit, last_write_wg_id}][{unit, wg_id}]
                      .emplace(distance, buffer, producer, consumer,
                               num_versions);
                }
              }
            }
          }
          // WAR/WAW: unit writes buffer, wait for all last readers
          // if there are no last readers, wait for last writer
          auto first_writes = unit->GetFirstAccessTasks(
              buffer, /*is_write=*/true, wg_id, SchedulePhase::kBody);
          if (!first_writes.empty()) {
            bool has_last_read = false;
            for (int last_wg = 0; last_wg < num_wgs; ++last_wg) {
              if (last_read_unit[last_wg] == nullptr)
                continue;
              has_last_read = true;
              for (auto *consumer : first_writes) {
                for (auto *producer : last_read_unit_tasks[last_wg]) {
                  if (IsRegisterBuffer(buffer) && wg_id != last_wg) {
                    // Skip cross-warpgroup dependency for register buffers
                    continue;
                  }
                  sync_infos[{last_read_unit[last_wg], last_wg}][{unit, wg_id}]
                      .emplace(distance, buffer, producer, consumer,
                               num_versions);
                }
              }
            }
            if (!has_last_read && last_write_unit != nullptr &&
                !waited_write_wgs[wg_id]) {
              for (auto *consumer : first_writes) {
                for (const auto &[last_write_wg_id, last_write_unit_tasks] :
                     last_write_unit_wg_tasks) {
                  for (auto *producer : last_write_unit_tasks) {
                    if (IsRegisterBuffer(buffer) && wg_id != last_write_wg_id) {
                      // Skip cross-warpgroup dependency for register buffers
                      continue;
                    }
                    sync_infos[{last_write_unit, last_write_wg_id}]
                              [{unit, wg_id}]
                                  .emplace(distance, buffer, producer, consumer,
                                           num_versions);
                  }
                }
              }
            }
          }
        }
        // Set status to avoid redundant dependencies for subsequent units
        for (const auto &buffer_access :
             unit->GetBufferAccessInfo(num_wgs, SchedulePhase::kBody)) {
          int wg_id = buffer_access.warpgroup_id;
          if (buffer_access.buffer != buffer)
            continue;
          if (!buffer_access.is_write) {
            waited_write_wgs[wg_id] = true;
            last_read_unit[wg_id] = nullptr;
          } else {
            for (int wg_id = 0; wg_id < num_wgs; ++wg_id) {
              last_read_unit[wg_id] = nullptr;
            }
            last_write_unit = nullptr;
          }
        }
        if (iter == 0) {
          // Update last_read info
          for (int wg = 0; wg < num_wgs; ++wg) {
            auto last_reads = unit->GetLastAccessTasks(
                buffer, /*is_write=*/false, wg, SchedulePhase::kBody);
            if (!last_reads.empty()) {
              last_read_unit[wg] = unit;
              last_read_unit_tasks[wg] = std::move(last_reads);
            }
          }
          // Update last_write info
          {
            std::vector<int> write_wg_ids;
            for (const auto &ba :
                 unit->GetBufferAccessInfo(num_wgs, SchedulePhase::kBody)) {
              if (ba.buffer == buffer && ba.is_write) {
                if (std::find(write_wg_ids.begin(), write_wg_ids.end(),
                              ba.warpgroup_id) == write_wg_ids.end())
                  write_wg_ids.push_back(ba.warpgroup_id);
              }
            }
            if (!write_wg_ids.empty()) {
              last_write_unit = unit;
              last_write_unit_wg_tasks.clear();
              for (int wg : write_wg_ids) {
                auto last_writes = unit->GetLastAccessTasks(
                    buffer, /*is_write=*/true, wg, SchedulePhase::kBody);
                if (!last_writes.empty())
                  last_write_unit_wg_tasks.emplace_back(wg,
                                                        std::move(last_writes));
              }
              for (int wg = 0; wg < num_wgs; ++wg)
                waited_write_wgs[wg] = false;
            }
          }
        }
      }
    }
  }
  return sync_infos;
}

static void InsertSynchronization(
    const std::vector<ScheduleUnit *> &units,
    const std::map<std::pair<ScheduleUnit *, int>,
                   std::map<std::pair<ScheduleUnit *, int>, std::set<SyncInfo>>>
        &sync_infos,
    int &next_barrier_id, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map,
    const std::vector<PrimExpr> &thread_count, LoopNestingInfo &loop_info) {
  int num_wgs = thread_count.size();
  // Initiate WGMMA tracking structures
  /*
  std::vector<int> wgmma_count(num_wgs, 0);
  std::vector<std::map<ScheduleUnit *, int>> wgmma_id(num_wgs);
  for (auto unit : units) {
    for (int wg_id = 0; wg_id < num_wgs; ++wg_id) {
      wgmma_id[wg_id][unit] = wgmma_count[wg_id];
    }
    if (unit->HasWGMMA() && unit->isInnerTask()) {
      int wg_id = static_cast<TaskNode *>(unit->child.get())->GetWarpgroupId();
      if (unit->GetSchedulePhase() == SchedulePhase::kBody) {
        ICHECK(0 <= wg_id && wg_id < num_wgs);
        ++wgmma_count[wg_id];
      } else {
        LOG(FATAL) << "WGMMA task without valid warpgroup id";
      }
    }
  }
  */
  // Insert synchronization statements based on sync_infos
  for (auto unit : units) {
    for (int wg_id = 0; wg_id < num_wgs; ++wg_id) {
      auto sync_it = sync_infos.find({unit, wg_id});
      int barrier_versions = 1;
      if (sync_it != sync_infos.end()) {
        for (const auto &[waiting_unit_info, sync_infos] : sync_it->second) {
          for (const auto &sync_info : sync_infos) {
            barrier_versions =
                std::max(barrier_versions, sync_info.buffer_versions);
          }
        }
      }
      Buffer barrier_buffer;
      // Handle single special task, such as TCGEN05 or TMA load, that requires
      // a barrier for itself.
      if (auto task = GetInnerTask(unit)) {
        int task_wg_id = task->GetWarpgroupId();
        if (task->is_TCGEN05() && task_wg_id == wg_id) {
          int barrier_id = next_barrier_id++;
          barrier_buffer = makeBarrierBuffer(
              1, "tcgen05_barrier_" + std::to_string(barrier_id),
              barrier_versions, barrier_buffers, barrier_map);
          PrimExpr version_index =
              indexmod(loop_info.CalculateIterationCount(), barrier_versions);
          PrimExpr mbar_expr = BufferLoad(barrier_buffer, {version_index});
          RewriteGemmMbar(task, mbar_expr);
          // TODO: need to change the lower of tcgen05_gemm to check if there is
          // already a arrive statement. Then we can manually insert the arrive
          // statement to deal with the case where the tcgen05_gemm is inside an
          // if condition.
          /*
          Stmt arrive_stmt =
              makeTcgen05MmaArrive(barrier_buffer, version_index);
          InsertStatementIntoScheduleUnit(unit, arrive_stmt, false, wg_id);
          */
        }
        if (task->HasTMALoad() && task_wg_id == wg_id) {
          int barrier_id = next_barrier_id++;
          barrier_buffer = makeBarrierBuffer(
              thread_count[wg_id], "tma_barrier_" + std::to_string(barrier_id),
              barrier_versions, barrier_buffers, barrier_map);
          PrimExpr version_index =
              indexmod(loop_info.CalculateIterationCount(), barrier_versions);
          PrimExpr mbar_expr = BufferLoad(barrier_buffer, {version_index});
          RewriteCopyMbar(task, mbar_expr);
          Stmt arrive_stmt = makeBarrierArrive(mbar_expr);
          InsertStatementIntoScheduleUnit(unit, arrive_stmt, false, wg_id);
        }
      }
      if (sync_it == sync_infos.end())
        continue;
      const auto &wait_map = sync_it->second;
      auto check_need_sync = [&](ScheduleUnit *waiting_unit, int waiting_wg_id,
                                 const SyncInfo &sync_info) {
        if (unit == waiting_unit)
          // Note: the logic here need some assumption.
          return false;
        if (wg_id != waiting_wg_id)
          return true;
        if (!sync_info.producer->UsesTMACore() &&
            !sync_info.producer->UsesTensorCore())
          return false;
        if (sync_info.producer->UsesTensorCore() &&
            sync_info.consumer->UsesTensorCore())
          return false;
        return true;
      };
      // Handle WGMMA synchronization
      {
        auto check_need_wgmma_sync = [&](ScheduleUnit *waiting_unit,
                                         int waiting_wg_id,
                                         const SyncInfo &sync_info) {
          return check_need_sync(waiting_unit, waiting_wg_id, sync_info) &&
                 sync_info.producer->is_WGMMA();
        };
        bool has_wgmma_sync = false;
        for (const auto &[waiting_unit_info, sync_infos] : wait_map) {
          auto [waiting_unit, waiting_wg_id] = waiting_unit_info;
          for (const auto &sync_info : sync_infos) {
            if (check_need_wgmma_sync(waiting_unit, waiting_wg_id, sync_info)) {
              has_wgmma_sync = true;
              break;
            }
          }
          if (has_wgmma_sync) {
            break;
          }
        }
        if (has_wgmma_sync) {
          bool different_wg_id = false;
          for (const auto &[waiting_unit_info, sync_infos] : wait_map) {
            auto [waiting_unit, waiting_wg_id] = waiting_unit_info;
            if (wg_id == waiting_wg_id) {
              continue;
            }
            for (const auto &sync_info : sync_infos) {
              if (check_need_wgmma_sync(waiting_unit, waiting_wg_id,
                                        sync_info)) {
                different_wg_id = true;
                break;
              }
            }
            if (different_wg_id) {
              break;
            }
          }
          if (!different_wg_id) {
            for (const auto &[waiting_unit_info, sync_infos] : wait_map) {
              auto [waiting_unit, waiting_wg_id] = waiting_unit_info;
              bool need_wait = false;
              for (const auto &sync_info : sync_infos) {
                if (check_need_wgmma_sync(waiting_unit, waiting_wg_id,
                                          sync_info)) {
                  need_wait = true;
                  break;
                }
              }
              if (need_wait) {
                Stmt wait_stmt =
                    Evaluate(Call(DataType::Handle(), wait_wgmma(), {0}));
                InsertStatementIntoScheduleUnit(waiting_unit, wait_stmt, true,
                                                wg_id);
              }
            }
          } else {
            Stmt wait_stmt =
                Evaluate(Call(DataType::Handle(), wait_wgmma(), {0}));
            InsertStatementIntoScheduleUnit(unit, wait_stmt, false, wg_id);
          }
        }
      }
      auto check_need_barrier = [&](ScheduleUnit *waiting_unit,
                                    int waiting_wg_id,
                                    const SyncInfo &sync_info) {
        return check_need_sync(waiting_unit, waiting_wg_id, sync_info) &&
               (wg_id != waiting_wg_id || !sync_info.producer->is_WGMMA());
      };
      bool need_barrier = false;
      for (const auto &[waiting_unit_info, sync_infos] : wait_map) {
        auto [waiting_unit, waiting_wg_id] = waiting_unit_info;
        for (const auto &sync_info : sync_infos) {
          if (check_need_barrier(waiting_unit, waiting_wg_id, sync_info)) {
            need_barrier = true;
            break;
          }
        }
        if (need_barrier) {
          break;
        }
      }
      if (!need_barrier)
        continue;
      if (!barrier_buffer.defined()) {
        // Note: the logic here assumes that we DO NOT need to wait for TMA
        // loads in this unit. If this assumption does not hold, we may need to
        // implement a more complex logic to synchronize.
        if (unit->HasTCGEN05()) {
          int barrier_id = next_barrier_id++;
          barrier_buffer = makeBarrierBuffer(
              1, "tcgen05_barrier_" + std::to_string(barrier_id),
              barrier_versions, barrier_buffers, barrier_map);
          PrimExpr version_index =
              indexmod(loop_info.CalculateIterationCount(), barrier_versions);
          Stmt arrive_stmt =
              makeTcgen05MmaArrive(barrier_buffer, version_index);
          InsertStatementIntoScheduleUnit(unit, arrive_stmt, false, wg_id);
        } else {
          int barrier_id = next_barrier_id++;
          barrier_buffer = makeBarrierBuffer(
              thread_count[wg_id], "barrier_" + std::to_string(barrier_id),
              barrier_versions, barrier_buffers, barrier_map);
          PrimExpr version_index =
              indexmod(loop_info.CalculateIterationCount(), barrier_versions);
          PrimExpr mbar_expr = BufferLoad(barrier_buffer, {version_index});
          Stmt arrive_stmt = makeBarrierArrive(mbar_expr);
          InsertStatementIntoScheduleUnit(unit, arrive_stmt, false, wg_id);
        }
      }
      // Add wait statements for all waiting units.
      for (const auto &[waiting_unit_info, sync_infos] : wait_map) {
        auto [waiting_unit, waiting_wg_id] = waiting_unit_info;
        int distance = 100;
        for (const auto &sync_info : sync_infos) {
          if (check_need_barrier(waiting_unit, waiting_wg_id, sync_info)) {
            distance = std::min(distance, sync_info.distance);
          }
        }
        if (distance < 100) {
          PrimExpr iteration = loop_info.CalculateIterationCount() - distance;
          PrimExpr version_index = indexmod(iteration, barrier_versions);
          PrimExpr mbar_expr = BufferLoad(barrier_buffer, {version_index});
          PrimExpr parity_expr =
              indexmod(indexdiv(iteration, barrier_versions), 2);
          Stmt wait_stmt = makeBarrierWait(mbar_expr, parity_expr);
          InsertStatementIntoScheduleUnit(waiting_unit, wait_stmt, true,
                                          waiting_wg_id);
        }
      }
    }
  }
}

static void
AnalyzeSequenceNodeBarriers(SequenceNode *seq, int &next_barrier_id,
                            std::vector<Buffer> &barrier_buffers,
                            Map<ObjectRef, ObjectRef> &barrier_map,
                            const std::vector<PrimExpr> &thread_count,
                            LoopNestingInfo &loop_info,
                            std::vector<MultiVersionBufferInfo> &buffer_infos,
                            Buffer neutral_sync_shared_barrier, bool is_root) {
  if (!seq)
    return;

  // Collect all units from the sequence
  std::vector<ScheduleUnit *> units;
  for (auto &child : seq->children) {
    auto unit = static_cast<ScheduleUnit *>(child.get());
    units.push_back(unit);
    if (GetInnerTask(unit) != nullptr) {
      // We will handle these units specially in InsertSynchronization, so we
      // skip it here.
      continue;
    }
    if (unit->child->IsSequence() || unit->child->IsControl() ||
        unit->child->IsIf()) {
      // If child is SequenceNode, ControlNode, or IfNode, recursively analyze
      // it
      AnalyzeAndInsertBarriers(
          unit->child.get(), next_barrier_id, barrier_buffers, barrier_map,
          thread_count, loop_info, buffer_infos, neutral_sync_shared_barrier);
    }
  }

  // Rewrite TMA load units to use tma_copy and neutral_sync_shared_barrier
  for (auto unit : units) {
    if (auto task = GetInnerTask(unit)) {
      if (task->HasTMALoad() &&
          task->GetSchedulePhase() == SchedulePhase::kPrologue) {
        PrimExpr barrier_load = BufferLoad(neutral_sync_shared_barrier, {0});
        RewriteCopyMbar(task, barrier_load);
      }
    }
  }

  // Analyze dependencies and insert synchronization
  auto sync_infos = GetSyncInfos(units, thread_count.size());
  InsertSynchronization(units, sync_infos, next_barrier_id, barrier_buffers,
                        barrier_map, thread_count, loop_info);

  // For the root, since we will insert kAutoScheduleSharedMemoryBoundary before
  // and after for-loop segments, we
  // naively add barriers at these positions to ensure synchronization.
  if (is_root) {
    int num_wgs = thread_count.size();
    for (const auto &unit : units) {
      if (!unit->child->IsControl())
        continue;
      {
        std::vector<Buffer> barrier_buffer(num_wgs);
        for (int wg_id = 0; wg_id < num_wgs; ++wg_id) {
          int barrier_id = next_barrier_id++;
          barrier_buffer[wg_id] = makeBarrierBuffer(
              thread_count[wg_id], "root_barrier_" + std::to_string(barrier_id),
              1, barrier_buffers, barrier_map);
        }
        for (int wg_id = 0; wg_id < num_wgs; ++wg_id) {
          for (int other_wg_id = 0; other_wg_id < num_wgs; ++other_wg_id) {
            if (wg_id == other_wg_id)
              continue;
            PrimExpr mbar_expr = BufferLoad(barrier_buffer[wg_id], {0});
            PrimExpr parity_expr = IntImm(DataType::Int(32), 0);
            Stmt wait_stmt = makeBarrierWait(mbar_expr, parity_expr);
            InsertStatementIntoScheduleUnit(unit, wait_stmt, true, other_wg_id);
          }
        }
        for (int wg_id = 0; wg_id < num_wgs; ++wg_id) {
          PrimExpr mbar_expr = BufferLoad(barrier_buffer[wg_id], {0});
          Stmt arrive_stmt = makeBarrierArrive(mbar_expr);
          InsertStatementIntoScheduleUnit(unit, arrive_stmt, true, wg_id);
        }
      }
      {
        std::vector<Buffer> barrier_buffer(num_wgs);
        for (int wg_id = 0; wg_id < num_wgs; ++wg_id) {
          int barrier_id = next_barrier_id++;
          barrier_buffer[wg_id] = makeBarrierBuffer(
              thread_count[wg_id], "root_barrier_" + std::to_string(barrier_id),
              1, barrier_buffers, barrier_map);
        }
        for (int wg_id = 0; wg_id < num_wgs; ++wg_id) {
          PrimExpr mbar_expr = BufferLoad(barrier_buffer[wg_id], {0});
          Stmt arrive_stmt = makeBarrierArrive(mbar_expr);
          InsertStatementIntoScheduleUnit(unit, arrive_stmt, false, wg_id);
        }
        for (int wg_id = 0; wg_id < num_wgs; ++wg_id) {
          for (int other_wg_id = 0; other_wg_id < num_wgs; ++other_wg_id) {
            if (wg_id == other_wg_id)
              continue;
            PrimExpr mbar_expr = BufferLoad(barrier_buffer[wg_id], {0});
            PrimExpr parity_expr = IntImm(DataType::Int(32), 0);
            Stmt wait_stmt = makeBarrierWait(mbar_expr, parity_expr);
            InsertStatementIntoScheduleUnit(unit, wait_stmt, false, other_wg_id);
          }
        }
      }
    }
  }
}

static void
AnalyzeControlNodeBarriers(ControlNode *ctrl, int &next_barrier_id,
                           std::vector<Buffer> &barrier_buffers,
                           Map<ObjectRef, ObjectRef> &barrier_map,
                           const std::vector<PrimExpr> &thread_count,
                           LoopNestingInfo &loop_info,
                           std::vector<MultiVersionBufferInfo> &buffer_infos,
                           Buffer neutral_sync_shared_barrier, bool is_root) {
  if (!ctrl || !ctrl->child)
    return;

  // Get loop information
  const ForNode *for_node = ctrl->control.get();
  if (!for_node)
    return;

  // Add this loop to nesting info
  loop_info.AddLoop(for_node);

  // Collect all units from the sequence
  ICHECK(ctrl->child->IsSequence());
  auto seq = static_cast<SequenceNode *>(ctrl->child.get());
  std::vector<ScheduleUnit *> units;
  for (auto &child : seq->children) {
    auto unit = static_cast<ScheduleUnit *>(child.get());
    units.push_back(unit);
    if (GetInnerTask(unit) != nullptr) {
      // We will handle these units specially in InsertSynchronization, so we
      // skip it here.
      continue;
    }
    if (unit->child->IsSequence() || unit->child->IsControl() ||
        unit->child->IsIf()) {
      // If child is SequenceNode, ControlNode, or IfNode, recursively analyze
      // it
      AnalyzeAndInsertBarriers(
          unit->child.get(), next_barrier_id, barrier_buffers, barrier_map,
          thread_count, loop_info, buffer_infos, neutral_sync_shared_barrier);
    }
  }

  // Sort units by stage
  // This matches the software pipelining order
  auto ordered_units = units;
  std::stable_sort(
      ordered_units.begin(), ordered_units.end(),
      [](ScheduleUnit *a, ScheduleUnit *b) { return a->stage > b->stage; });

  // Detect multi-version buffers and create new buffers for them
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      multi_buffer;
  std::unordered_map<Buffer, int, ObjectPtrHash, ObjectPtrEqual>
      buffer_num_versions;
  int num_wgs = thread_count.size();
  for (const auto &unit : ordered_units) {
    for (const auto &buffer_access :
         unit->GetBufferAccessInfo(num_wgs, SchedulePhase::kBody)) {
      auto &buffer = buffer_access.buffer;
      if (!ctrl->multi_buffering_buffers.count(buffer))
        continue;
      for (const auto &other_unit : ordered_units) {
        if (unit == other_unit)
          continue;
        int distance = unit->child->GetStartTime() + unit->child->GetLatency() -
                       other_unit->child->GetStartTime();
        if (distance <= 0)
          continue;
        distance = (distance - 1) / ctrl->GetIIperIter() + 1;
        for (const auto &other_buffer_access :
             other_unit->GetBufferAccessInfo(num_wgs, SchedulePhase::kBody)) {
          auto &other_buffer = other_buffer_access.buffer;
          if (!buffer.same_as(other_buffer))
            continue;
          if (buffer_access.is_write || other_buffer_access.is_write) {
            auto &num_versions = buffer_num_versions[buffer];
            num_versions = std::max(num_versions, distance);
          }
        }
      }
    }
  }
  for (auto &region : ctrl->GetWriteRegions()) {
    auto &buffer = region.get()->buffer;
    if (!ctrl->multi_buffering_buffers.count(buffer))
      continue;
    if (multi_buffer.find(buffer) != multi_buffer.end())
      continue;
    auto it = buffer_num_versions.find(buffer);
    if (it == buffer_num_versions.end())
      continue;
    int num_versions = it->second;
    if (num_versions == 1)
      continue;
    auto new_buffer = RewriteAllocBuffer(buffer, num_versions);
    multi_buffer[buffer] = new_buffer;
    buffer_infos.emplace_back(buffer, num_versions, new_buffer);
  }

  // Rewrite BufferLoad/BufferStore in TaskNode stmts for multi-version
  // buffers
  PrimExpr iteration = loop_info.CalculateIterationCount();

  // Recursively rewrite all TaskNode stmts
  RewriteTaskNodeBuffers(ctrl, multi_buffer, iteration);

  // Analyze dependencies and insert synchronization
  auto sync_infos = GetSyncInfos(ordered_units, thread_count.size(),
                                 buffer_num_versions, true);
  InsertSynchronization(units, sync_infos, next_barrier_id, barrier_buffers,
                        barrier_map, thread_count, loop_info);

  // Remove this loop from nesting info when exiting
  loop_info.PopLoop();
}
} // namespace tl
} // namespace tvm
