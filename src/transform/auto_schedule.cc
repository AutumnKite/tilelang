/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_schedule.cc
 * \brief AutoSchedule pass for TileLang
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../layout/layout.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/gemm.h"
#include "../target/utils.h"
#include "./common/attr.h"
#include "./common/collector.h"
#include "auto_schedule.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

// Mutator to update thread extent in AttrStmt nodes
// Used after warpgroup partition to double thread extent
class ThreadExtentUpdater : public StmtExprMutator {
public:
  explicit ThreadExtentUpdater(PrimExpr updated_extent)
      : updated_thread_extent_(updated_extent) {}

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      auto thread_iv_ = Downcast<IterVar>(op->node);
      if (thread_iv_->thread_tag == "threadIdx.x") {
        // Visit the body first (to update any references)
        AttrStmt attr_stmt =
            Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));

        // Update the thread extent

        // Create new IterVar with updated domain
        Range new_dom =
            Range::FromMinExtent(thread_iv_->dom->min, updated_thread_extent_);

        // Update the AttrStmt with new IterVar and value
        thread_iv_.CopyOnWrite()->dom = new_dom;
        attr_stmt.CopyOnWrite()->node = thread_iv_;
        attr_stmt.CopyOnWrite()->value = updated_thread_extent_;

        // Clear the saved reference
        thread_iv_ = {};

        return attr_stmt;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

private:
  PrimExpr updated_thread_extent_;
  IterVar thread_iv_;
};

// Visitor to extract the body of tilelang_root block
class TilelangRootBodyExtractor : public StmtVisitor {
public:
  Stmt body;

  void VisitStmt_(const BlockNode *op) override {
    if (op->name_hint == "tilelang_root") {
      body = op->body;
      return; // Don't visit children
    }
    StmtVisitor::VisitStmt_(op);
  }
};

// Detect multiple kernel launches in the PrimFunc body.
// In tilelang, when multiple T.Kernel() blocks are used, the IR structure is:
//   root block body:
//     AttrStmt(tl.assume, ...)
//       AttrStmt(tl.assume, ...)
//         SeqStmt [
//           AttrStmt(blockIdx.x, thread_extent, ..., kernel1_subtree),
//           AttrStmt(blockIdx.x, thread_extent, ..., kernel2_subtree),
//         ]
// Each kernel subtree contains its own launch_threads and tilelang_root block.
// This class finds that SeqStmt and returns each child as a separate kernel.
class MultiKernelDetector {
public:
  static bool Detect(const Stmt &func_body, std::vector<Stmt> &kernel_stmts,
                     Stmt &prefix_wrapper) {
    std::vector<Stmt> stmts;
    const Stmt *inner = &func_body;

    // Peel through root block -> BlockRealize
    if (const auto *br = inner->as<BlockRealizeNode>()) {
      inner = &br->block->body;
    }

    // Peel through AttrStmt(tl.assume, ...) chains
    while (const auto *attr = inner->as<AttrStmtNode>()) {
      if (attr->attr_key != "tl.assume")
        break;
      inner = &attr->body;
    }

    // Check if we have a SeqStmt with multiple children that each contain
    // a launch_thread (thread_extent)
    const auto *seq = inner->as<SeqStmtNode>();
    if (!seq || seq->seq.size() < 2)
      return false;

    int kernel_count = 0;
    for (const auto &child : seq->seq) {
      if (ContainsLaunchThread(child)) {
        kernel_count++;
      }
    }

    if (kernel_count < 2)
      return false;

    for (const auto &child : seq->seq) {
      kernel_stmts.push_back(child);
    }
    return true;
  }

private:
  static bool ContainsLaunchThread(const Stmt &stmt) {
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == tir::attr::thread_extent) {
        return true;
      }
    }
    return false;
  }
};

// Mutator that replaces the inner SeqStmt (inside root block -> tl.assume
// chain) with a new body. Used to reassemble multi-kernel results.
class InnerSeqStmtReplacer : public StmtMutator {
public:
  explicit InnerSeqStmtReplacer(Stmt new_inner) : new_inner_(new_inner) {}

  Stmt VisitStmt_(const BlockRealizeNode *op) override {
    auto new_block_body = this->VisitStmt(op->block->body);
    if (new_block_body.same_as(op->block->body))
      return GetRef<Stmt>(op);
    auto new_block =
        Block(op->block->iter_vars, op->block->reads, op->block->writes,
              op->block->name_hint, new_block_body, op->block->init,
              op->block->alloc_buffers, op->block->match_buffers,
              op->block->annotations);
    return BlockRealize(op->iter_values, op->predicate, new_block);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == "tl.assume") {
      auto new_body = this->VisitStmt(op->body);
      if (new_body.same_as(op->body))
        return GetRef<Stmt>(op);
      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    }
    return GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) override { return new_inner_; }

private:
  Stmt new_inner_;
};

// Mutator to replace the body of tilelang_root block
class TilelangRootBodyReplacer : public StmtMutator {
public:
  explicit TilelangRootBodyReplacer(Stmt new_body) : new_body_(new_body) {}

  Stmt VisitStmt_(const BlockNode *op) override {
    auto block = GetRef<Block>(op);
    if (op->name_hint == "tilelang_root") {
      // Keep all block attributes but replace the body
      return Block(op->iter_vars, op->reads, op->writes, op->name_hint,
                   new_body_, op->init, op->alloc_buffers, op->match_buffers,
                   op->annotations);
    }
    return StmtMutator::VisitStmt_(op);
  }

private:
  Stmt new_body_;
};

// Visitor to build IRStructure from TIR statements
class IRStructureBuilder : public StmtVisitor {
public:
  std::shared_ptr<IRStructure> Build(const Stmt &stmt, int64_t thread_count = 1,
                                     Target target = Target()) {
    thread_count_ = thread_count;
    target_ = target;
    VisitStmt(stmt);
    if (!root_) {
      LOG(WARNING)
          << "IRStructureBuilder: root_ is null after visiting statement. "
          << "This may indicate an unhandled statement type.";
      // Return an empty TaskNode as fallback
      auto task_node = std::make_shared<TaskNode>();
      task_node->stmts.push_back(stmt);
      return task_node;
    }
    return std::move(root_);
  }

protected:
  void VisitStmt_(const SeqStmtNode *op) override {
    auto seq_node = std::make_shared<SequenceNode>();

    for (size_t i = 0; i < op->seq.size(); i++) {
      VisitStmt(op->seq[i]);
      if (root_) {
        seq_node->children.push_back(std::move(root_));
      }
    }
    root_ = std::move(seq_node);
  }

  void VisitStmt_(const ForNode *op) override {
    // Determine if this is a sequential or parallel for
    if (op->kind == ForKind::kSerial) {
      // Sequential For -> ControlNode
      auto control_node = std::make_shared<ControlNode>();
      control_node->control = GetRef<For>(op);
      control_node->task = std::make_shared<TaskNode>();
      control_node->task->SetWarpgroupId(kWarpgroupBroadcast);
      control_node->task->stmts.push_back(
          For(op->loop_var, op->min, op->extent, op->kind, Evaluate(0),
              op->thread_binding, op->annotations, op->step, op->span));
      AnalyzeResourceUsage(Evaluate(op->min), control_node->task.get(), true);
      AnalyzeResourceUsage(Evaluate(op->extent), control_node->task.get(),
                           true);
      if (op->step.defined())
        AnalyzeResourceUsage(Evaluate(GetRef<PrimExpr>(op->step.get())),
                             control_node->task.get(), true);

      // Process the loop body
      VisitStmt(op->body);
      if (root_) {
        control_node->child = std::move(root_);
      } else {
      }

      root_ = std::move(control_node);
    } else {
      // Parallel For -> TaskNode
      auto task_node = std::make_shared<TaskNode>();
      task_node->stmts.push_back(GetRef<Stmt>(op));

      // Analyze the loop body for resource usage)
      AnalyzeResourceUsage(op->body, task_node.get());
      AnalyzeResourceUsage(Evaluate(op->min), task_node.get(), true);
      AnalyzeResourceUsage(Evaluate(op->extent), task_node.get(), true);
      if (op->step.defined())
        AnalyzeResourceUsage(Evaluate(GetRef<PrimExpr>(op->step.get())),
                             task_node.get(), true);

      root_ = std::move(task_node);
    }
  }

  void VisitStmt_(const EvaluateNode *op) override {
    // Evaluate statement (usually a Call) -> TaskNode
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze the expression for resource usage
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const BufferStoreNode *op) override {
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const IfThenElseNode *op) override {
    // If statement -> IfNode with independently schedulable branches
    auto if_node = std::make_shared<IfNode>();
    if_node->condition = op->condition;

    // Create task for condition expression resource analysis
    auto cond_task = std::make_shared<TaskNode>();
    cond_task->stmts.push_back(Evaluate(op->condition));
    AnalyzeMemoryExpr(op->condition, cond_task.get());
    AnalyzeResourceUsage(Evaluate(op->condition), cond_task.get(), true);
    if_node->task = std::move(cond_task);

    // Recursively build then branch
    VisitStmt(op->then_case);
    if (root_) {
      if_node->then_child = std::move(root_);
    }

    // Recursively build else branch (if present)
    if (op->else_case) {
      VisitStmt(op->else_case.value());
      if (root_) {
        if_node->else_child = std::move(root_);
      }
    }

    // Latency = max of both branches
    int64_t then_latency =
        if_node->then_child ? if_node->then_child->GetLatency() : 0;
    int64_t else_latency =
        if_node->else_child ? if_node->else_child->GetLatency() : 0;
    if_node->SetLatency(std::max(then_latency, else_latency));

    root_ = std::move(if_node);
  }

  void VisitStmt_(const LetStmtNode *op) override {
    // Wrapper statement -> WrapperNode
    auto wrapper_node = std::make_shared<WrapperNode>();
    wrapper_node->wrapper = GetRef<Stmt>(op);
    auto task_node = std::make_shared<TaskNode>();
    task_node->SetWarpgroupId(kWarpgroupBroadcast);
    task_node->stmts.push_back(GetLetDecl(op));
    AnalyzeResourceUsage(GetLetDecl(op), task_node.get());
    wrapper_node->task = std::move(task_node);

    // Process the wrapperbody
    VisitStmt(op->body);
    if (root_) {
      wrapper_node->child = std::move(root_);
    }

    root_ = std::move(wrapper_node);
  }

  void VisitStmt_(const AttrStmtNode *op) override {
    // Wrapper statement -> WrapperNode
    auto wrapper_node = std::make_shared<WrapperNode>();
    wrapper_node->wrapper = GetRef<Stmt>(op);
    auto task_node = std::make_shared<TaskNode>();
    task_node->SetWarpgroupId(kWarpgroupBroadcast);
    task_node->stmts.push_back(GetAttrDecl(op));
    AnalyzeResourceUsage(GetAttrDecl(op), task_node.get());
    wrapper_node->task = std::move(task_node);

    // Process the wrapperbody
    VisitStmt(op->body);
    if (root_) {
      wrapper_node->child = std::move(root_);
    }

    root_ = std::move(wrapper_node);
  }

  void VisitStmt_(const WhileNode *op) override {
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));

    // Analyze condition and body for resource usage
    AnalyzeResourceUsage(Evaluate(op->condition), task_node.get());
    AnalyzeResourceUsage(op->body, task_node.get());

    root_ = std::move(task_node);
  }

  void VisitStmt_(const BlockNode *op) override {
    // All blocks are treated as TaskNode
    // Note: tilelang_root block should have been extracted by
    // TilelangRootBodyExtractor If we encounter it here, it means we're
    // processing the entire function body (not extracted), which should only
    // happen when there's no tilelang_root block
    auto task_node = std::make_shared<TaskNode>();
    task_node->stmts.push_back(GetRef<Stmt>(op));
    AnalyzeResourceUsage(GetRef<Stmt>(op), task_node.get());
    root_ = std::move(task_node);
  }

private:
  std::shared_ptr<IRStructure> root_;
  int64_t thread_count_ = 1;
  Target target_;

  void AnalyzeResourceUsage(const Stmt &stmt, TaskNode *task_node,
                            bool only_variables = false) {
    // Recursively analyze statements to determine resource usage
    struct ResourceAnalyzer : public StmtExprVisitor {
      TaskNode *task_node;
      bool found_tma{false};
      bool found_tensor{false};
      bool found_cuda{false};

      bool found_tma_load{false};

      // Tensor Core shape information (multiple shapes possible)
      struct TensorCoreShape {
        int64_t m;
        int64_t n;
        int64_t k;
        TensorCoreShape(int64_t m = 0, int64_t n = 0, int64_t k = 0)
            : m(m), n(n), k(k) {}
      };
      std::vector<TensorCoreShape> tensor_core_shapes;

      // GemmInst: the resolved tensor core instruction (single, asserted
      // shared)
      GemmInst gemm_inst{GemmInst::kMMA};
      bool has_gemm_inst{false};

      // Target and block_size for GemmInst determination
      Target target;
      int64_t block_size;

      ResourceAnalyzer(TaskNode *node, Target target = Target(),
                       int64_t block_size = 128)
          : task_node(node), target(target), block_size(block_size) {}

      void VisitExpr_(const CallNode *op) override {
        // Check for specific TileLang operations
        static const auto copy_op = Op::Get("tl.tileop.copy");
        static const auto gemm_op = Op::Get("tl.tileop.gemm");
        static const auto wgmma_gemm_op = Op::Get("tl.tileop.wgmma_gemm");
        static const auto tcgen05_gemm_op = Op::Get("tl.tileop.tcgen05_gemm");
        static const auto reduce_op = Op::Get("tl.tileop.reduce");
        static const auto fill_op = Op::Get("tl.tileop.fill");
        static const auto region_op = Op::Get("tl.tileop.region");

        // Try to get operation name for logging
        std::string op_name = "unknown";
        if (const auto *op_ptr = op->op.as<OpNode>()) {
          op_name = op_ptr->name;
        }

        // Check if this is a TMA copy operation
        static const auto tma_copy_op = Op::Get("tl.tileop.tma_copy");
        static const auto async_copy_op = Op::Get("tl.tileop.async_copy");

        bool is_copy_like = op->op.same_as(copy_op) ||
                            op->op.same_as(tma_copy_op) ||
                            op->op.same_as(async_copy_op);

        if (is_copy_like) {
          Copy copy_obj(op->args, op->annotations);
          const CopyNode *copy = copy_obj.get();

          if (copy->GetIsAsyncCopy()) {
            // T.async_copy() — cp.async path, never TMA.
          } else if (copy->GetIsTmaCopy()) {
            // Explicit T.tma_copy(): only valid global->shared TMA loads
            // are producers; TMA stores stay on the consumer side.
            arith::Analyzer ana;
            if (copy->CheckBulkLoad(target, &ana, /*check_last_dim=*/false)) {
              found_tma = true;
              found_tma_load = true;
            }
          } else {
            // Generic T.copy(): check if TMA is possible.
            arith::Analyzer ana;
            if (!copy->GetDisableTMA()) {
              if (copy->CheckBulkLoad(target, &ana, /*check_last_dim=*/true)) {
                found_tma = true;
                found_tma_load = true;
              }
              if (copy->CheckBulkStore(target, &ana, /*check_last_dim=*/true)) {
                found_tma = true;
              }
            }
          }
        } else if (op->op.same_as(gemm_op) || op->op.same_as(wgmma_gemm_op) ||
                   op->op.same_as(tcgen05_gemm_op)) {
          found_tensor = true;

          int64_t m = op->args[5].as<IntImmNode>()->value;
          int64_t n = op->args[6].as<IntImmNode>()->value;
          int64_t k = op->args[7].as<IntImmNode>()->value;
          tensor_core_shapes.emplace_back(m, n, k);

          // Determine the final GemmInst using GemmPyNode::getGemmInst
          if (target.defined()) {
            Gemm gemm(op->args);
            GemmInst inst =
                gemm->getGemmInst(static_cast<int>(block_size), target);
            ICHECK(!has_gemm_inst || gemm_inst == inst)
                << "All gemm operations in a task must use the same GemmInst, "
                << "but got " << GemmInstToString(gemm_inst) << " and "
                << GemmInstToString(inst);
            gemm_inst = inst;
            has_gemm_inst = true;
          }
        } else if (op->op.same_as(reduce_op) || op->op.same_as(fill_op)) {
          // Reduce and fill operations use CUDA core
          found_cuda = true;
        } else if (op->op.same_as(region_op)) {
          // Handle tl.tileop.region call for memory access analysis
          // args[0] = buffer (BufferLoad), args[1] = access_type (1: read, 2:
          // write, 3: read/write) args[2..] = extents
          if (op->args.size() >= 2) {
            // Extract access type
            if (const auto *access_int = op->args[1].as<IntImmNode>()) {
              int access_type = access_int->value;
              // For now, just mark as CUDA operation (memory access)
              found_cuda = true;
              // TODO: Extract buffer region and add to task_node->read_regions
              // or write_regions BufferLoad buffer_load =
              // Downcast<BufferLoad>(op->args[0]); Construct BufferRegion from
              // buffer_load and extents if (access_type == 1 || access_type ==
              // 3) task_node->read_regions.push_back(region); if (access_type
              // == 2 || access_type == 3)
              // task_node->write_regions.push_back(region);
            }
          }
        } else {
          // Check for other known operations that use CUDA core
          // For now, assume any other call is a basic computation
          found_cuda = true;
        }

        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const AddNode *op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const SubNode *op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const MulNode *op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const DivNode *op) override {
        found_cuda = true;
        StmtExprVisitor::VisitExpr_(op);
      }
    };

    ResourceAnalyzer analyzer(task_node, target_, thread_count_);
    analyzer(stmt);

    if (!only_variables) {
      // Set task node flags based on what was found
      if (analyzer.found_tma) {
        task_node->SetUsesTMACore(true);
        if (analyzer.found_tma_load) {
          task_node->SetHasTMALoad(true);
        }
      }
      if (analyzer.found_tensor) {
        task_node->SetUsesTensorCore(true);
        // Set Tensor Core shape information if available
        for (const auto &shape : analyzer.tensor_core_shapes) {
          if (shape.m > 0 && shape.n > 0 && shape.k > 0) {
            task_node->AddTensorCoreShape(shape.m, shape.n, shape.k);
          }
        }
        // Set GemmInst information
        if (analyzer.has_gemm_inst) {
          task_node->SetGemmInst(analyzer.gemm_inst);
        }
      }
      // If neither TMA nor Tensor core was used, and CUDA operations were
      // found, set CUDA core flag
      if (!analyzer.found_tma && !analyzer.found_tensor) {
        task_node->SetUsesCUDACore(true);
      }
    }

    // Analyze memory access regions
    MemoryAccessDetector memory_detector;
    memory_detector.Analyze(stmt);
    std::vector<BufferRegion> read_regions = memory_detector.GetReadRegions();
    std::vector<BufferRegion> write_regions = memory_detector.GetWriteRegions();
    std::vector<Var> read_vars = memory_detector.GetReadVars();
    std::vector<Var> write_vars = memory_detector.GetWriteVars();

    // Merge with existing regions (avoid duplicates)
    for (const auto &region : read_regions) {
      task_node->AddReadRegion(region);
    }

    for (const auto &region : write_regions) {
      task_node->AddWriteRegion(region);
    }

    for (const auto &var : read_vars) {
      task_node->AddReadVar(var);
    }

    for (const auto &var : write_vars) {
      task_node->AddWriteVar(var);
    }

    // Estimate latency and initiation interval for this task
    LatencyEstimator latency_estimator(target_);
    latency_estimator.SetThreadCount(thread_count_);
    latency_estimator.Estimate(task_node);
  }

  void AnalyzeMemoryExpr(const PrimExpr &expr, TaskNode *task_node) {
    // Analyze memory access regions
    MemoryAccessDetector memory_detector;
    memory_detector.Analyze(expr);
    std::vector<BufferRegion> read_regions = memory_detector.GetReadRegions();
    std::vector<BufferRegion> write_regions = memory_detector.GetWriteRegions();

    // Merge with existing regions (avoid duplicates)
    for (const auto &region : read_regions) {
      task_node->AddReadRegion(region);
    }

    for (const auto &region : write_regions) {
      task_node->AddWriteRegion(region);
    }

    // Estimate latency and initiation interval for this task
    LatencyEstimator latency_estimator(target_);
    latency_estimator.SetThreadCount(thread_count_);
    latency_estimator.Estimate(task_node);
  }
};

Stmt ReNestLetStmts(const Stmt &stmt);

// Result of scheduling a single kernel segment
struct ScheduledKernelResult {
  Stmt scheduled_body;
  std::vector<Buffer> barrier_buffers;
  Map<ObjectRef, ObjectRef> barrier_map;
  std::vector<MultiVersionBufferInfo> buffer_infos;
  std::vector<Buffer> duplicated_fragment_buffers;
  PrimExpr updated_thread_extent;
  bool did_warpgroup_partition{false};
};

// Schedule a single kernel body (the logic previously inlined in AutoSchedule).
// This handles IRStructure building, ScheduleUnit building, barrier analysis,
// and warpgroup partition for one kernel.
static ScheduledKernelResult
ScheduleSingleKernel(const Stmt &kernel_body, IterVar thread_var, Target target,
                     const WarpSpecializeConfig &config, bool aggressive,
                     bool enable_epi) {
  ScheduledKernelResult result;

  // Calculate thread count for latency estimation
  int64_t latency_thread_count = 1;
  if (thread_var.defined() && thread_var->dom.defined()) {
    PrimExpr thread_extent = thread_var->dom->extent;
    if (const int64_t *extent_ptr = as_const_int(thread_extent)) {
      latency_thread_count = *extent_ptr;
      if (latency_thread_count < 1)
        latency_thread_count = 1;
    }
  }

  // Build IRStructure from the body to schedule
  IRStructureBuilder builder;
  auto ir_structure = builder.Build(kernel_body, latency_thread_count, target);

  // Print the built IRStructure with all statements
  ICHECK(ir_structure) << "IRStructure is null (empty body?)";

  // Build ScheduleUnits from IRStructure
  ScheduleUnitBuilder unit_builder;
  if (thread_var.defined()) {
    unit_builder.SetThreadVar(thread_var);
  } else {
    LOG(FATAL) << "Could not find thread index variable, warpgroup "
                  "partition will use default";
  }
  unit_builder.SetWarpSpecializeConfig(config);
  unit_builder.SetSharedMemoryLimit(GetSharedMemoryLimit(target));

  std::vector<PrimExpr> thread_count;
  if (!aggressive) {
    thread_count = unit_builder.NaiveBuild(ir_structure);
  } else {
    thread_count = unit_builder.Build(ir_structure);
  }

  if (!config.enable_warpgroup_partition) {
    result.scheduled_body =
        ConvertIRStructureToStmt(ir_structure.get(), enable_epi);
    result.scheduled_body = StripUnusedLetStmts(result.scheduled_body);
    result.did_warpgroup_partition = false;
    return result;
  }

  // Print the modified summary view
  // PrintIRStructure(ir_structure.get());

  // Analyze buffer dependencies and insert barriers before warpgroup
  // partition
  int next_barrier_id = 1;
  LoopNestingInfo loop_info;
  PrimExpr updated_thread_extent = std::accumulate(
      thread_count.begin() + 1, thread_count.end(), thread_count[0]);
  result.updated_thread_extent = updated_thread_extent;
  Buffer neutral_sync_shared_barrier =
      makeBarrierBuffer(updated_thread_extent, "neutral_sync_shared_barrier", 1,
                        result.barrier_buffers, result.barrier_map);
  AnalyzeAndInsertBarriers(ir_structure.get(), next_barrier_id,
                           result.barrier_buffers, result.barrier_map,
                           thread_count, loop_info, result.buffer_infos,
                           neutral_sync_shared_barrier);

  // Print the modified summary view
  // PrintIRStructure(ir_structure.get());

  // Apply warpgroup partition to entire IRStructure
  result.scheduled_body = ApplyWarpgroupPartitionToIRStructure(
      ir_structure.get(), thread_var, result.barrier_buffers,
      result.barrier_map, enable_epi, thread_count, config,
      neutral_sync_shared_barrier, result.duplicated_fragment_buffers);
  result.scheduled_body = StripUnusedLetStmts(result.scheduled_body);
  result.did_warpgroup_partition = true;
  return result;
}

// Helper: add barrier buffers and barrier_map to the tilelang_root block
static Stmt AddBarrierBuffersToRoot(const Stmt &body,
                                    const std::vector<Buffer> &barrier_buffers,
                                    Map<ObjectRef, ObjectRef> &barrier_map) {
  class TilelangRootAllocBufferAdder : public StmtMutator {
  public:
    explicit TilelangRootAllocBufferAdder(
        const std::vector<Buffer> &buffers_to_add,
        Map<ObjectRef, ObjectRef> &barrier_map)
        : buffers_to_add_(buffers_to_add), barrier_map_(barrier_map) {}

    Stmt VisitStmt_(const BlockNode *op) override {
      auto block = GetRef<Block>(op);
      if (op->name_hint == "tilelang_root") {
        // Combine existing alloc_buffers with new buffers
        Array<Buffer> new_alloc_buffers = op->alloc_buffers;
        for (const auto &buffer : buffers_to_add_) {
          new_alloc_buffers.push_back(buffer);
        }
        auto new_annotations = op->annotations;
        new_annotations.Set("barrier_init", barrier_map_);
        // Create new block with updated alloc_buffers
        return Block(op->iter_vars, op->reads, op->writes, op->name_hint,
                     op->body, op->init, new_alloc_buffers, op->match_buffers,
                     new_annotations);
      }
      return StmtMutator::VisitStmt_(op);
    }

  private:
    std::vector<Buffer> buffers_to_add_;
    Map<ObjectRef, ObjectRef> &barrier_map_;
  };
  TilelangRootAllocBufferAdder adder(barrier_buffers, barrier_map);
  return adder(body);
}

// The main pass function
tvm::transform::Pass AutoSchedule(const bool enable_epi) {
  using namespace tir::transform;
  auto pass_func =
      [enable_epi](PrimFunc func, const IRModule &mod,
                   const tvm::transform::PassContext &ctx) -> PrimFunc {
    // Get target from PrimFunc attribute for GemmInst determination
    auto target_opt = func->GetAttr<Target>(tvm::attr::kTarget);
    Target target;
    if (target_opt.defined()) {
      target = target_opt.value();
    }
    auto config = GetWarpSpecializeConfig(target);

    // Check if aggressive auto-schedule is enabled
    bool aggressive =
        ctx->GetConfig<Bool>(kEnableAggressiveAutoSchedule, Bool(true)).value();

    // Detect multiple kernel launches in the PrimFunc body.
    // When multiple T.Kernel() blocks are used, the IR has a SeqStmt
    // containing separate kernel subtrees, each with its own tilelang_root.
    std::vector<Stmt> kernel_stmts;
    Stmt prefix_wrapper;
    bool is_multi_kernel =
        MultiKernelDetector::Detect(func->body, kernel_stmts, prefix_wrapper);

    if (!is_multi_kernel) {
      // --- Single-kernel path (original behavior) ---
      // Extract the body of tilelang_root block if it exists
      TilelangRootBodyExtractor extractor;
      extractor(func->body);
      Stmt body_to_schedule;

      if (extractor.body.defined()) {
        body_to_schedule = extractor.body;
      } else {
        LOG(FATAL);
        body_to_schedule = func->body;
      }

      // Get thread index variable for warpgroup partition
      // First try to get from body_to_schedule, if not found, try from the
      // entire function body
      IterVar thread_var = ThreadTagChecker::GetThreadVar(body_to_schedule);
      if (!thread_var.defined()) {
        thread_var = ThreadTagChecker::GetThreadVar(func->body);
      }

      auto kr = ScheduleSingleKernel(body_to_schedule, thread_var, target,
                                     config, aggressive, enable_epi);

      // If we extracted from tilelang_root block, replace the body
      Stmt final_body;
      TilelangRootBodyReplacer replacer(kr.scheduled_body);
      final_body = replacer(func->body);

      if (kr.did_warpgroup_partition) {
        // Apply thread extent update if warpgroup partition was applied
        // (sm_90 only)
        if (config.enable_thread_extend) {
          ThreadExtentUpdater extent_updater(kr.updated_thread_extent);
          final_body = extent_updater(final_body);
        }
        // Add barrier buffers to tilelang_root block's alloc_buffers
        if (!kr.barrier_buffers.empty() ||
            !kr.duplicated_fragment_buffers.empty()) {
          std::vector<Buffer> all_alloc_buffers = kr.barrier_buffers;
          all_alloc_buffers.insert(all_alloc_buffers.end(),
                                   kr.duplicated_fragment_buffers.begin(),
                                   kr.duplicated_fragment_buffers.end());
          final_body = AddBarrierBuffersToRoot(final_body, all_alloc_buffers,
                                               kr.barrier_map);
        }
        // Apply multi-version alloc_buffer rewrite if needed
        if (!kr.buffer_infos.empty()) {
          final_body = RewriteAllocBuffers(final_body, kr.buffer_infos);
        }
      }

      final_body = ReNestLetStmts(final_body);

      // Create a new PrimFunc with the updated body
      auto new_func = PrimFunc(func->params, final_body, func->ret_type,
                               func->buffer_map, func->attrs);
      return new_func;
    }

    // --- Multi-kernel path ---
    // Each kernel_stmts[i] is a complete kernel subtree:
    //   AttrStmt(blockIdx.x) -> ... -> AttrStmt(threadIdx.x) ->
    //     BlockRealize("tilelang_root") -> body
    // Schedule each independently and reassemble with shared memory
    // boundary markers between them.
    Array<Stmt> combined_stmts;

    for (size_t i = 0; i < kernel_stmts.size(); ++i) {
      Stmt kernel_subtree = kernel_stmts[i];

      // Extract the tilelang_root body from this kernel subtree
      TilelangRootBodyExtractor extractor;
      extractor(kernel_subtree);

      if (!extractor.body.defined()) {
        // Not a schedulable kernel (no tilelang_root), pass through
        if (!combined_stmts.empty()) {
          combined_stmts.push_back(
              AttrStmt(Integer(0), attr::kAutoScheduleSharedMemoryBoundary, 0,
                       Evaluate(0)));
        }
        combined_stmts.push_back(kernel_subtree);
        continue;
      }

      Stmt body_to_schedule = extractor.body;

      // Get thread index variable for this kernel
      IterVar thread_var = ThreadTagChecker::GetThreadVar(kernel_subtree);
      if (!thread_var.defined()) {
        // Fallback: pass through without scheduling
        if (!combined_stmts.empty()) {
          combined_stmts.push_back(
              AttrStmt(Integer(0), attr::kAutoScheduleSharedMemoryBoundary, 0,
                       Evaluate(0)));
        }
        combined_stmts.push_back(kernel_subtree);
        continue;
      }

      // Schedule this kernel independently
      auto kr = ScheduleSingleKernel(body_to_schedule, thread_var, target,
                                     config, aggressive, enable_epi);

      // Replace the tilelang_root body in this kernel subtree
      Stmt scheduled_subtree;
      {
        TilelangRootBodyReplacer replacer(kr.scheduled_body);
        scheduled_subtree = replacer(kernel_subtree);
      }

      if (kr.did_warpgroup_partition) {
        // Apply thread extent update if warpgroup partition was applied
        // (sm_90 only)
        if (config.enable_thread_extend) {
          ThreadExtentUpdater extent_updater(kr.updated_thread_extent);
          scheduled_subtree = extent_updater(scheduled_subtree);
        }
        // Add barrier buffers to this kernel's tilelang_root block
        if (!kr.barrier_buffers.empty() ||
            !kr.duplicated_fragment_buffers.empty()) {
          std::vector<Buffer> all_alloc_buffers = kr.barrier_buffers;
          all_alloc_buffers.insert(all_alloc_buffers.end(),
                                   kr.duplicated_fragment_buffers.begin(),
                                   kr.duplicated_fragment_buffers.end());
          scheduled_subtree = AddBarrierBuffersToRoot(
              scheduled_subtree, all_alloc_buffers, kr.barrier_map);
        }
        // Apply multi-version alloc_buffer rewrite if needed
        if (!kr.buffer_infos.empty()) {
          scheduled_subtree =
              RewriteAllocBuffers(scheduled_subtree, kr.buffer_infos);
        }
      }

      // Insert shared memory boundary between kernel segments
      if (!combined_stmts.empty()) {
        combined_stmts.push_back(
            AttrStmt(Integer(0), attr::kAutoScheduleSharedMemoryBoundary, 0,
                     Evaluate(0)));
      }
      combined_stmts.push_back(scheduled_subtree);
    }

    // Reassemble: replace the inner SeqStmt in the PrimFunc body with the
    // new combined statements
    Stmt new_inner;
    if (combined_stmts.size() == 1) {
      new_inner = combined_stmts[0];
    } else {
      new_inner = SeqStmt(combined_stmts);
    }

    InnerSeqStmtReplacer seq_replacer(new_inner);
    Stmt final_body = seq_replacer(func->body);

    final_body = ReNestLetStmts(final_body);

    // Create a new PrimFunc with the updated body
    auto new_func = PrimFunc(func->params, final_body, func->ret_type,
                             func->buffer_map, func->attrs);
    return new_func;
  };

  return CreatePrimFuncPass(pass_func, 0, "tl.AutoSchedule", {});
}

// Re-write LetStmt to nest them properly
// Example transformation:
//   SeqStmt {
//     let x = 42 { Evaluate(0) }     // standalone, empty body
//     let y = x+1 { Evaluate(0) }    // standalone, empty body
//     compute(x, y)                   // actual work
//     store(result)
//   }
// becomes:
//   let x = 42 {
//     let y = x+1 {
//       SeqStmt {
//         compute(x, y)
//         store(result)
//       }
//     }
//   }
class LetStmtNester : public StmtMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) override {
    Array<Stmt> stmts;
    for (const auto &stmt : op->seq) {
      stmts.push_back(this->VisitStmt(stmt));
    }

    Array<Stmt> flat_stmts;
    for (const auto &stmt : stmts) {
      if (const auto *inner_seq = stmt.as<SeqStmtNode>()) {
        for (const auto &inner_stmt : inner_seq->seq) {
          flat_stmts.push_back(inner_stmt);
        }
      } else {
        flat_stmts.push_back(stmt);
      }
    }
    stmts = flat_stmts;

    for (int i = static_cast<int>(stmts.size()) - 2; i >= 0; --i) {
      if (const auto *let = stmts[i].as<LetStmtNode>()) {
        if (IsEmptyBody(let->body)) {
          Stmt absorbed_body = CollectRemaining(stmts, i + 1);
          stmts = TruncateAndReplace(
              stmts, i, LetStmt(let->var, let->value, absorbed_body));
        }
      } else if (const auto *attr = stmts[i].as<AttrStmtNode>()) {
        if (IsEmptyBody(attr->body)) {
          Stmt absorbed_body = CollectRemaining(stmts, i + 1);
          stmts = TruncateAndReplace(
              stmts, i,
              AttrStmt(attr->node, attr->attr_key, attr->value, absorbed_body));
        }
      }
    }

    if (stmts.empty())
      return Evaluate(0);
    if (stmts.size() == 1)
      return stmts[0];

    return SeqStmt(stmts);
  }

private:
  // Check if a statement body is Evaluate(0) — the empty placeholder
  static bool IsEmptyBody(const Stmt &stmt) {
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *imm = eval->value.as<IntImmNode>()) {
        return imm->value == 0;
      }
    }
    return false;
  }

  // Collect stmts[start .. end) into a single Stmt
  static Stmt CollectRemaining(const Array<Stmt> &stmts, int start) {
    int n = static_cast<int>(stmts.size());
    if (start >= n) {
      return Evaluate(0);
    }
    if (start == n - 1) {
      return stmts[start];
    }
    Array<Stmt> remaining;
    for (int j = start; j < n; ++j) {
      remaining.push_back(stmts[j]);
    }
    return SeqStmt(remaining);
  }

  // Keep stmts[0..index), replace stmts[index] with new_stmt,
  // discard everything after (already absorbed into new_stmt body)
  static Array<Stmt> TruncateAndReplace(const Array<Stmt> &stmts, int index,
                                        Stmt new_stmt) {
    Array<Stmt> result;
    for (int j = 0; j < index; ++j) {
      result.push_back(stmts[j]);
    }
    result.push_back(new_stmt);
    return result;
  }
};

// Recursively flatten all nested SeqStmt nodes throughout the IR tree.
// This must run before LetStmtNester so that every SeqStmt it encounters
// is already flat, preventing incorrect LetStmt/AttrStmt absorption across
// nested SeqStmt boundaries.
class SeqStmtFlattener : public StmtMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) override {
    // First, recursively visit children.
    Array<Stmt> visited;
    for (const auto &s : op->seq) {
      visited.push_back(this->VisitStmt(s));
    }
    // Then flatten: if any child is itself a SeqStmt, inline its children.
    Array<Stmt> flat;
    std::function<void(const Stmt &)> Flatten = [&](const Stmt &s) {
      if (const auto *inner = s.as<SeqStmtNode>()) {
        for (const auto &inner_s : inner->seq) {
          Flatten(inner_s);
        }
      } else {
        flat.push_back(s);
      }
    };
    for (const auto &s : visited) {
      Flatten(s);
    }
    if (flat.empty())
      return Evaluate(0);
    if (flat.size() == 1)
      return flat[0];
    return SeqStmt(flat);
  }
};

Stmt ReNestLetStmts(const Stmt &stmt) {
  SeqStmtFlattener flattener;
  Stmt flat = flattener(stmt);
  LetStmtNester nester;
  return nester(flat);
}

// StmtMutator to rewrite alloc_buffers in Block nodes
namespace {

bool LayoutShapesEqual(const Array<PrimExpr> &lhs, const Array<PrimExpr> &rhs,
                       arith::Analyzer *analyzer) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!analyzer->CanProveEqual(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

// Expand an annotated Layout so its InputShape matches the multi-versioned
// buffer shape by prepending the leading "num_versions" dim(s).
Layout ExpandAnnotatedLayoutForMultiVersionedBuffer(const Layout &layout,
                                                    const Buffer &old_buffer,
                                                    const Buffer &new_buffer) {
  if (!layout.defined() ||
      new_buffer->shape.size() <= old_buffer->shape.size()) {
    return Layout();
  }

  arith::Analyzer analyzer;
  if (!LayoutShapesEqual(layout->InputShape(), old_buffer->shape, &analyzer)) {
    return Layout();
  }

  size_t leading_ndim = new_buffer->shape.size() - old_buffer->shape.size();
  Array<PrimExpr> trailing_shape;
  Array<PrimExpr> leading_shape;
  for (size_t i = 0; i < leading_ndim; ++i) {
    leading_shape.push_back(new_buffer->shape[i]);
  }
  for (size_t i = 0; i < old_buffer->shape.size(); ++i) {
    trailing_shape.push_back(new_buffer->shape[leading_ndim + i]);
  }
  if (!LayoutShapesEqual(trailing_shape, old_buffer->shape, &analyzer)) {
    return Layout();
  }

  return layout->Expand(leading_shape);
}

// Walk the block's layout_map annotation and expand any entries whose buffer
// has been multi-versioned so downstream LayoutInference sees a matching shape.
bool UpdateExpandedLayoutMapForRemappedAllocs(
    const std::vector<std::pair<Buffer, Buffer>> &remapped_allocs,
    Map<String, ffi::Any> *annotations) {
  if (remapped_allocs.empty() || !annotations->count(attr::kLayoutMap)) {
    return false;
  }

  auto layout_map_ref = annotations->Get(attr::kLayoutMap);
  if (!layout_map_ref.has_value()) {
    return false;
  }
  auto layout_map = layout_map_ref.value().as<Map<Var, Layout>>();
  if (!layout_map.has_value()) {
    return false;
  }

  Map<Var, Layout> updated_layout_map = layout_map.value();
  std::unordered_set<const VarNode *> visited;
  bool changed = false;
  for (const auto &[old_buffer, new_buffer] : remapped_allocs) {
    if (!visited.insert(old_buffer->data.get()).second ||
        !updated_layout_map.count(old_buffer->data)) {
      continue;
    }
    Layout layout = updated_layout_map[old_buffer->data];
    Layout expanded = ExpandAnnotatedLayoutForMultiVersionedBuffer(
        layout, old_buffer, new_buffer);
    if (!expanded.defined()) {
      continue;
    }
    updated_layout_map.Set(old_buffer->data, expanded);
    changed = true;
  }

  if (changed) {
    annotations->Set(attr::kLayoutMap, updated_layout_map);
  }
  return changed;
}

} // namespace

class AllocBufferRewriter : public StmtMutator {
public:
  AllocBufferRewriter(const std::vector<MultiVersionBufferInfo> &buffer_infos)
      : buffer_infos_(buffer_infos) {
    // Create mapping from original buffer to new buffer
    for (const auto &info : buffer_infos_) {
      buffer_remap_[info.buffer] = info.new_buffer;
    }
  }

private:
  Stmt VisitStmt_(const BlockNode *op) override {
    Stmt new_body = this->VisitStmt(op->body);

    // Check if we need to update alloc_buffers
    bool needs_update = false;
    Array<Buffer> new_alloc_buffers;
    std::vector<std::pair<Buffer, Buffer>> remapped_allocs;

    for (auto buffer : op->alloc_buffers) {
      auto it = buffer_remap_.find(buffer);
      if (it != buffer_remap_.end()) {
        new_alloc_buffers.push_back(it->second);
        remapped_allocs.emplace_back(buffer, it->second);
        needs_update = true;
      } else {
        new_alloc_buffers.push_back(buffer);
      }
    }

    auto new_block = CopyOnWrite(op);
    new_block->body = new_body;
    if (needs_update) {
      new_block->alloc_buffers = new_alloc_buffers;
      UpdateExpandedLayoutMapForRemappedAllocs(remapped_allocs,
                                               &new_block->annotations);
    }
    return Stmt(new_block);
  }

  const std::vector<MultiVersionBufferInfo> &buffer_infos_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;
};

// Main function to rewrite alloc_buffers
Stmt RewriteAllocBuffers(
    const Stmt &stmt, const std::vector<MultiVersionBufferInfo> &buffer_infos) {
  if (buffer_infos.empty()) {
    return stmt;
  }

  AllocBufferRewriter rewriter(buffer_infos);
  return rewriter(stmt);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AutoSchedule", AutoSchedule);
}

} // namespace tl
} // namespace tvm
