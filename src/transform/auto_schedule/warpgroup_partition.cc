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
 * \file warpgroup_partition.cc
 * \brief Warpgroup partition and IRStructure-to-Stmt conversion for TileLang
 * AutoSchedule
 */

#include "./warpgroup_partition.h"

#include "../auto_schedule.h"
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
#include <functional>
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

#include "../../op/builtin.h"
#include "../../target/utils.h"
#include "../common/attr.h"
#include "../common/collector.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

class LetStmtVarRenamer : public StmtExprMutator {
public:
  explicit LetStmtVarRenamer(const std::string &suffix) : suffix_(suffix) {}

  Stmt Rename(Stmt stmt) {
    CollectLetVars(stmt);
    if (var_remap_.empty())
      return stmt;
    return VisitStmt(stmt);
  }

private:
  void CollectLetVars(const Stmt &stmt) {
    class Collector : public StmtExprVisitor {
    public:
      explicit Collector(Map<Var, Var> &remap, const std::string &suffix)
          : remap_(remap), suffix_(suffix) {}
      void VisitStmt_(const LetStmtNode *op) final {
        remap_.Set(op->var, op->var.copy_with_suffix(suffix_));
        StmtExprVisitor::VisitStmt_(op);
      }
      Map<Var, Var> &remap_;
      const std::string &suffix_;
    };
    Collector c(var_remap_, suffix_);
    c(stmt);
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    auto it = var_remap_.find(op->var);
    Var new_var = it != var_remap_.end() ? (*it).second : op->var;
    PrimExpr new_value = VisitExpr(op->value);
    Stmt new_body = VisitStmt(op->body);
    return LetStmt(new_var, new_value, new_body, op->span);
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    auto it = var_remap_.find(var);
    if (it != var_remap_.end()) {
      return (*it).second;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  std::string suffix_;
  Map<Var, Var> var_remap_;
};

static Stmt RenameLetStmtVars(Stmt stmt, const std::string &suffix) {
  return LetStmtVarRenamer(suffix).Rename(std::move(stmt));
}

// Mutator that replaces references to selected Buffers with their duplicates.
class BufferRemapMutator : public StmtExprMutator {
public:
  explicit BufferRemapMutator(const Map<Buffer, Buffer> &buffer_remap)
      : buffer_remap_(buffer_remap) {
    for (const auto &kv : buffer_remap_) {
      var_to_new_buffer_.Set(kv.first->data, kv.second);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = buffer_remap_.find(load->buffer);
    if (it == buffer_remap_.end())
      return std::move(load);
    auto *n = load.CopyOnWrite();
    n->buffer = (*it).second;
    return std::move(load);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = buffer_remap_.find(store->buffer);
    if (it == buffer_remap_.end())
      return std::move(store);
    auto *n = store.CopyOnWrite();
    n->buffer = (*it).second;
    return std::move(store);
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    auto it = var_to_new_buffer_.find(var);
    if (it != var_to_new_buffer_.end()) {
      return (*it).second->data;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  BufferRegion RemapRegion(const BufferRegion &region) const {
    auto it = buffer_remap_.find(region->buffer);
    if (it == buffer_remap_.end())
      return region;
    return BufferRegion((*it).second, region->region);
  }

  Var RemapVar(const Var &var) const {
    auto it = var_to_new_buffer_.find(var);
    if (it == var_to_new_buffer_.end())
      return var;
    return (*it).second->data;
  }

private:
  const Map<Buffer, Buffer> &buffer_remap_;
  Map<Var, Buffer> var_to_new_buffer_;
};

// Collect all local / local.var / local.fragment Buffers written by
// broadcast tasks so each warpgroup gets its own private copy.
static void CollectBroadcastFragmentBuffersImpl(
    const IRStructure *node, std::unordered_set<const BufferNode *> &seen,
    std::vector<Buffer> &out) {
  if (!node)
    return;
  if (node->IsTask()) {
    auto task = static_cast<const TaskNode *>(node);
    if (IsWarpgroupBroadcast(task->GetWarpgroupId())) {
      for (const auto &region : task->GetWriteRegions()) {
        if (IsRegisterRegion(region) &&
            (IsFragmentBuffer(region->buffer) ||
             IsLocalBuffer(region->buffer, /*allow_var=*/true))) {
          if (!seen.count(region->buffer.get())) {
            seen.insert(region->buffer.get());
            out.push_back(region->buffer);
          }
        }
      }
    }
  } else if (node->IsSequence()) {
    for (const auto &child :
         static_cast<const SequenceNode *>(node)->children) {
      CollectBroadcastFragmentBuffersImpl(child.get(), seen, out);
    }
  } else if (node->IsControl()) {
    auto ctrl = static_cast<const ControlNode *>(node);
    CollectBroadcastFragmentBuffersImpl(ctrl->task.get(), seen, out);
    CollectBroadcastFragmentBuffersImpl(ctrl->child.get(), seen, out);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(node);
    CollectBroadcastFragmentBuffersImpl(wrapper->task.get(), seen, out);
    CollectBroadcastFragmentBuffersImpl(wrapper->child.get(), seen, out);
  } else if (node->IsScheduleUnit()) {
    CollectBroadcastFragmentBuffersImpl(
        static_cast<const ScheduleUnit *>(node)->child.get(), seen, out);
  } else if (node->IsIf()) {
    auto if_node = static_cast<const IfNode *>(node);
    CollectBroadcastFragmentBuffersImpl(if_node->task.get(), seen, out);
    CollectBroadcastFragmentBuffersImpl(if_node->then_child.get(), seen, out);
    CollectBroadcastFragmentBuffersImpl(if_node->else_child.get(), seen, out);
  }
}

static std::vector<Buffer>
CollectBroadcastFragmentBuffers(const IRStructure *root) {
  std::vector<Buffer> result;
  std::unordered_set<const BufferNode *> seen;
  CollectBroadcastFragmentBuffersImpl(root, seen, result);
  return result;
}

static Buffer DuplicateFragmentBuffer(const Buffer &buffer,
                                      const std::string &suffix) {
  Type new_type = buffer->data->type_annotation;
  if (IsFragmentBuffer(buffer)) {
    const auto *ptr_type = buffer->data->type_annotation.as<PointerTypeNode>();
    ICHECK(ptr_type);
    new_type = PointerType(ptr_type->element_type, "local");
  }
  Var new_var(buffer->data->name_hint + suffix, new_type);
  return Buffer(new_var, buffer->dtype, buffer->shape, buffer->strides,
                buffer->elem_offset, buffer->name + suffix,
                buffer->data_alignment, buffer->offset_factor,
                buffer->buffer_type);
}

static void ApplyBufferRemapToTask(TaskNode *task,
                                   BufferRemapMutator &mutator) {
  for (size_t i = 0; i < task->stmts.size(); ++i) {
    task->stmts[i] = mutator(task->stmts[i]);
  }
  auto read_regions = task->GetReadRegions();
  for (auto &r : read_regions)
    r = mutator.RemapRegion(r);
  task->SetReadRegions(read_regions);
  auto write_regions = task->GetWriteRegions();
  for (auto &r : write_regions)
    r = mutator.RemapRegion(r);
  task->SetWriteRegions(write_regions);
  auto read_vars = task->GetReadVars();
  for (auto &v : read_vars)
    v = mutator.RemapVar(v);
  task->SetReadVars(read_vars);
  auto write_vars = task->GetWriteVars();
  for (auto &v : write_vars)
    v = mutator.RemapVar(v);
  task->SetWriteVars(write_vars);
}

bool IsLetDeclTask(const TaskNode *task) {
  return task->stmts.size() == 1 && task->stmts[0].as<LetStmtNode>() != nullptr;
}

// Helper: check if an IRStructure node is a LetDecl task (or a ScheduleUnit
// wrapping one)
bool IsLetDeclNode(const IRStructure *node) {
  if (!node)
    return false;
  if (node->IsTask()) {
    return IsLetDeclTask(static_cast<const TaskNode *>(node));
  }
  if (node->IsScheduleUnit()) {
    auto unit = static_cast<const ScheduleUnit *>(node);
    return unit->child && unit->child->IsTask() &&
           IsLetDeclTask(static_cast<const TaskNode *>(unit->child.get()));
  }
  return false;
}

// Helper: check if an IRStructure subtree contains any LetDecl tasks
bool ContainsLetDecl(const IRStructure *node) {
  if (!node)
    return false;
  if (IsLetDeclNode(node))
    return true;
  if (node->IsSequence()) {
    auto seq = static_cast<const SequenceNode *>(node);
    for (const auto &child : seq->children) {
      if (ContainsLetDecl(child.get()))
        return true;
    }
  } else if (node->IsControl()) {
    auto ctrl = static_cast<const ControlNode *>(node);
    return ContainsLetDecl(ctrl->child.get());
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(node);
    return ContainsLetDecl(wrapper->child.get());
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<const ScheduleUnit *>(node);
    return ContainsLetDecl(unit->child.get());
  } else if (node->IsIf()) {
    auto if_node = static_cast<const IfNode *>(node);
    if (ContainsLetDecl(if_node->then_child.get()))
      return true;
    if (if_node->else_child && ContainsLetDecl(if_node->else_child.get()))
      return true;
  }
  return false;
}

// Helper function to clone IRStructure with warpgroup filter.
std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id,
                                    Map<Var, PrimExpr> &var_remap,
                                    Map<Buffer, Buffer> &buffer_remap) {
  if (!node)
    return nullptr;

  auto apply_buffer_remap_stmt = [&](Stmt s) -> Stmt {
    if (buffer_remap.empty())
      return s;
    BufferRemapMutator m(buffer_remap);
    return m(std::move(s));
  };
  auto apply_buffer_remap_expr = [&](PrimExpr e) -> PrimExpr {
    if (buffer_remap.empty())
      return e;
    BufferRemapMutator m(buffer_remap);
    return m(std::move(e));
  };
  auto apply_buffer_remap_task = [&](TaskNode *ct) {
    if (buffer_remap.empty())
      return;
    BufferRemapMutator m(buffer_remap);
    ApplyBufferRemapToTask(ct, m);
  };

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);
    if (task->GetSchedulePhase() != SchedulePhase::kBody) {
      auto new_task = std::make_shared<TaskNode>();
      return new_task;
    }

    // LetDecl tasks are always included in every warp group clone.
    // Create a fresh variable copy so the two warp groups use different names.
    if (IsLetDeclTask(task)) {
      const auto *let = task->stmts[0].as<LetStmtNode>();
      auto new_var = let->var.copy_with_suffix("");
      // Substitute previously renamed variables in the value expression.
      PrimExpr new_value =
          var_remap.empty() ? let->value : Substitute(let->value, var_remap);
      new_value = apply_buffer_remap_expr(new_value);
      var_remap.Set(let->var, new_var);
      auto new_task = std::make_shared<TaskNode>();
      new_task->stmts.push_back(LetStmt(new_var, new_value, Evaluate(0)));
      return new_task;
    }

    // Non-LetDecl tasks: only include if warp group matches
    if (!node->containWarpgroupId(warpgroup_id))
      return nullptr;
    auto cloned = task->Clone();
    // Substitute renamed LetDecl variables in task statements
    if (!var_remap.empty()) {
      auto ct = static_cast<TaskNode *>(cloned.get());
      for (size_t i = 0; i < ct->stmts.size(); ++i) {
        ct->stmts[i] = Substitute(ct->stmts[i], var_remap);
      }
    }
    apply_buffer_remap_task(static_cast<TaskNode *>(cloned.get()));
    return cloned;
  } else if (node->IsSequence()) {
    // A SequenceNode is included if it contains the target warp group
    // OR if it contains LetDecl tasks (which are always needed).
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto seq = static_cast<SequenceNode *>(node);
    auto new_seq = std::make_shared<SequenceNode>();
    for (const auto &child : seq->children) {
      auto new_child = CloneIRStructureWithWarpgroupFilter(
          child.get(), warpgroup_id, var_remap, buffer_remap);
      if (new_child) {
        new_seq->children.push_back(std::move(new_child));
      }
    }
    if (new_seq->children.empty())
      return nullptr;
    return new_seq;
  } else if (node->IsControl()) {
    // A ControlNode is included if it contains the target warp group
    // OR if it contains LetDecl tasks.
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto ctrl = static_cast<ControlNode *>(node);
    auto new_ctrl = std::make_shared<ControlNode>();
    For new_for = ctrl->control;
    auto new_loop_var = ctrl->control->loop_var.copy_with_suffix("");
    new_for.CopyOnWrite()->loop_var = new_loop_var;
    var_remap.Set(ctrl->control->loop_var, new_loop_var);
    new_for.CopyOnWrite()->min =
        apply_buffer_remap_expr(Substitute(ctrl->control->min, var_remap));
    new_for.CopyOnWrite()->extent =
        apply_buffer_remap_expr(Substitute(ctrl->control->extent, var_remap));
    if (ctrl->control->step.has_value()) {
      new_for.CopyOnWrite()->step = apply_buffer_remap_expr(
          Substitute(ctrl->control->step.value(), var_remap));
    }
    new_ctrl->control = new_for;
    // Clone the task and apply var_remap so each warpgroup gets its own copy
    // with correctly renamed LetDecl variables.
    if (ctrl->task) {
      auto cloned_task =
          std::static_pointer_cast<TaskNode>(ctrl->task->Clone());
      if (!var_remap.empty()) {
        for (size_t i = 0; i < cloned_task->stmts.size(); ++i) {
          cloned_task->stmts[i] = Substitute(cloned_task->stmts[i], var_remap);
        }
      }
      apply_buffer_remap_task(cloned_task.get());
      new_ctrl->task = std::move(cloned_task);
    }
    new_ctrl->SetPromote(ctrl->hasPromote());
    new_ctrl->child = CloneIRStructureWithWarpgroupFilter(
        ctrl->child.get(), warpgroup_id, var_remap, buffer_remap);
    return new_ctrl;
  } else if (node->IsWrapper()) {
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto wrapper = static_cast<WrapperNode *>(node);
    auto new_wrapper = std::make_shared<WrapperNode>();
    // Apply var_remap to the wrapper statement so that renamed LetDecl
    // variables are correctly substituted in LetStmt values / AttrStmt values.
    new_wrapper->wrapper = var_remap.empty()
                               ? wrapper->wrapper
                               : Substitute(wrapper->wrapper, var_remap);
    new_wrapper->wrapper = apply_buffer_remap_stmt(new_wrapper->wrapper);
    new_wrapper->child = CloneIRStructureWithWarpgroupFilter(
        wrapper->child.get(), warpgroup_id, var_remap, buffer_remap);
    return new_wrapper;
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    bool child_is_let_decl = IsLetDeclNode(unit->child.get());

    // Include the ScheduleUnit if the child is a LetDecl or the warp group
    // matches.
    if (!child_is_let_decl && !node->containWarpgroupId(warpgroup_id))
      return nullptr;

    auto new_unit = std::make_shared<ScheduleUnit>();
    new_unit->stage = unit->stage;
    new_unit->child = CloneIRStructureWithWarpgroupFilter(
        unit->child.get(), warpgroup_id, var_remap, buffer_remap);

    if (!child_is_let_decl) {
      // Copy before/after for the target warp group
      new_unit->before[warpgroup_id] = unit->before[warpgroup_id];
      new_unit->after[warpgroup_id] = unit->after[warpgroup_id];
      // Substitute renamed LetDecl variables in before/after stmts
      if (!var_remap.empty()) {
        for (auto &s : new_unit->before[warpgroup_id]) {
          s = Substitute(s, var_remap);
        }
        for (auto &s : new_unit->after[warpgroup_id]) {
          s = Substitute(s, var_remap);
        }
      }
      for (auto &s : new_unit->before[warpgroup_id]) {
        s = apply_buffer_remap_stmt(s);
      }
      for (auto &s : new_unit->after[warpgroup_id]) {
        s = apply_buffer_remap_stmt(s);
      }
    }
    return new_unit;
  } else if (node->IsIf()) {
    if (!node->containWarpgroupId(warpgroup_id) && !ContainsLetDecl(node))
      return nullptr;
    auto if_node = static_cast<IfNode *>(node);
    auto new_if = std::make_shared<IfNode>();
    new_if->condition = var_remap.empty()
                            ? if_node->condition
                            : Substitute(if_node->condition, var_remap);
    new_if->condition = apply_buffer_remap_expr(new_if->condition);
    if (if_node->task) {
      auto cloned_task =
          std::static_pointer_cast<TaskNode>(if_node->task->Clone());
      if (!var_remap.empty()) {
        for (size_t i = 0; i < cloned_task->stmts.size(); ++i) {
          cloned_task->stmts[i] = Substitute(cloned_task->stmts[i], var_remap);
        }
      }
      apply_buffer_remap_task(cloned_task.get());
      new_if->task = std::move(cloned_task);
    }
    new_if->then_child = CloneIRStructureWithWarpgroupFilter(
        if_node->then_child.get(), warpgroup_id, var_remap, buffer_remap);
    if (if_node->else_child) {
      new_if->else_child = CloneIRStructureWithWarpgroupFilter(
          if_node->else_child.get(), warpgroup_id, var_remap, buffer_remap);
    }
    // Return nullptr if both branches are empty
    if (!new_if->then_child && !new_if->else_child)
      return nullptr;
    return new_if;
  }
  LOG(FATAL);
  return nullptr;
}

std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id,
                                    Map<Var, PrimExpr> &var_remap) {
  Map<Buffer, Buffer> buffer_remap;
  return CloneIRStructureWithWarpgroupFilter(node, warpgroup_id, var_remap,
                                             buffer_remap);
}

std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id) {
  Map<Var, PrimExpr> var_remap;
  Map<Buffer, Buffer> buffer_remap;
  return CloneIRStructureWithWarpgroupFilter(node, warpgroup_id, var_remap,
                                             buffer_remap);
}

// For each child of a root SequenceNode, apply
// CloneIRStructureWithWarpgroupFilter individually.
std::vector<std::shared_ptr<IRStructure>>
CloneIRStructureChildrenWithWarpgroupFilter(SequenceNode *root_seq,
                                            int warpgroup_id,
                                            Map<Var, PrimExpr> &var_remap,
                                            Map<Buffer, Buffer> &buffer_remap) {
  std::vector<std::shared_ptr<IRStructure>> result;
  result.reserve(root_seq->children.size());
  for (const auto &child : root_seq->children) {
    result.push_back(CloneIRStructureWithWarpgroupFilter(
        child.get(), warpgroup_id, var_remap, buffer_remap));
  }
  return result;
}

std::vector<std::shared_ptr<IRStructure>>
CloneIRStructureChildrenWithWarpgroupFilter(SequenceNode *root_seq,
                                            int warpgroup_id,
                                            Map<Var, PrimExpr> &var_remap) {
  Map<Buffer, Buffer> buffer_remap;
  return CloneIRStructureChildrenWithWarpgroupFilter(root_seq, warpgroup_id,
                                                     var_remap, buffer_remap);
}

// Post-pass for the finished auto-schedule Stmt: drop any LetStmt whose bound
// variable is not referenced in its body, provided the bound value expression
// is pure (no side-effects beyond `kReadState`).
class UnusedLetStmtStripper : public StmtExprMutator {
public:
  Stmt VisitStmt_(const LetStmtNode *op) final {
    Stmt new_body = this->VisitStmt(op->body);
    PrimExpr new_value = this->VisitExpr(op->value);

    auto body_uses_var =
        UsesVar(new_body, [&](const VarNode *v) { return v == op->var.get(); });
    bool value_is_pure = SideEffect(new_value) <= CallEffectKind::kReadState;

    if (!body_uses_var && value_is_pure) {
      return new_body;
    }
    if (new_body.same_as(op->body) && new_value.same_as(op->value)) {
      return GetRef<Stmt>(op);
    }
    return LetStmt(op->var, new_value, new_body, op->span);
  }
};

Stmt StripUnusedLetStmts(const Stmt &stmt) {
  UnusedLetStmtStripper stripper;
  return stripper(stmt);
}

class SimtCopyDetector : public StmtExprVisitor {
public:
  static bool Detect(const Stmt &stmt) {
    SimtCopyDetector detector;
    detector.VisitStmt(stmt);
    return detector.has_simt_copy_;
  }

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    auto scope =
        runtime::StorageScope::Create(GetPtrStorageScope(op->buffer->data));
    if (scope.to_string() != "global") {
      has_simt_copy_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool has_simt_copy_{false};
};

// Detects whether `stmt` already carries any of the signals that downstream
// AnnotateWarpGroupRegAlloc (see src/transform/annotate_warp_group_reg_alloc.cc
// SetMaxNRegCollector) treats as "caller already decided register allocation".
// If any of these signals is present, the auto-schedule warp-group partitioner
// must skip its own (240/24-style) set_max_nreg injection so the inner choice
// is preserved.  The three honored signals are:
//   1. an explicit tl::set_max_nreg() call,
//   2. an explicit tl::no_set_max_nreg() opt-out sentinel,
//   3. an AttrStmt with key attr::kCustomWarpSpecialization.
class InnerNRegDecisionDetector : public StmtExprVisitor {
public:
  static bool Detect(const Stmt &stmt) {
    InnerNRegDecisionDetector detector;
    detector.VisitStmt(stmt);
    return detector.has_decision_;
  }

private:
  void VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(tl::set_max_nreg()) ||
          call->op.same_as(tl::no_set_max_nreg())) {
        has_decision_ = true;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == attr::kCustomWarpSpecialization) {
      has_decision_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool has_decision_{false};
};

Stmt ConvertIRStructureToStmt(IRStructure *structure,
                              const bool outer_enable_epi) {
  if (!structure) {
    return Evaluate(0);
  }

  if (structure->IsTask()) {
    auto task = static_cast<TaskNode *>(structure);
    if (task->stmts.empty()) {
      return Evaluate(0);
    } else if (task->stmts.size() == 1) {
      return task->stmts[0];
    } else {
      return SeqStmt(task->stmts);
    }
  } else if (structure->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(structure);
    std::vector<Stmt> stmts;
    for (const auto &child : seq->children) {
      auto unit = static_cast<ScheduleUnit *>(child.get());
      for (auto &[_, before] : unit->before) {
        for (auto &stmt : before) {
          stmts.push_back(stmt);
        }
      }
      Stmt child_stmt =
          ConvertIRStructureToStmt(unit->child.get(), outer_enable_epi);
      stmts.push_back(child_stmt);
      for (auto &[_, after] : unit->after) {
        for (auto &stmt : after) {
          stmts.push_back(stmt);
        }
      }
    }
    auto flattened = SeqStmt::Flatten(stmts);
    return flattened;
  } else if (structure->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(structure);
    Var loop_var = ctrl->control->loop_var;
    PrimExpr loop_start = ctrl->control->min;
    PrimExpr loop_extent = ctrl->control->extent;
    PrimExpr loop_step = ctrl->control->step.has_value()
                             ? ctrl->control->step.value()
                             : IntImm(DataType::Int(32), 1);
    int min_stages = 100, max_stages = -1;
    if (ctrl->child->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(ctrl->child.get());
      for (auto &child : seq->children) {
        auto unit = static_cast<ScheduleUnit *>(child.get());
        min_stages = std::min(min_stages, unit->stage);
        max_stages = std::max(max_stages, unit->stage);
      }
    }
    if (!ctrl->hasPromote() || !ctrl->child->IsSequence() ||
        min_stages == max_stages) {
      std::vector<Stmt> stmts;
      if (ctrl->child->IsScheduleUnit()) {
        auto unit = static_cast<ScheduleUnit *>(ctrl->child.get());
        for (auto &[_, before] : unit->before) {
          for (auto &stmt : before) {
            stmts.push_back(stmt);
          }
        }
        stmts.push_back(
            ConvertIRStructureToStmt(unit->child.get(), outer_enable_epi));
        for (auto &[_, after] : unit->after) {
          for (auto &stmt : after) {
            stmts.push_back(stmt);
          }
        }
      } else if (ctrl->child->IsSequence()) {
        auto seq = static_cast<SequenceNode *>(ctrl->child.get());
        for (auto &child : seq->children) {
          ICHECK(child->IsScheduleUnit());
          auto unit = static_cast<ScheduleUnit *>(child.get());
          for (auto &[_, before] : unit->before) {
            for (auto &stmt : before) {
              stmts.push_back(stmt);
            }
          }
          stmts.push_back(
              ConvertIRStructureToStmt(unit->child.get(), outer_enable_epi));
          for (auto &[_, after] : unit->after) {
            for (auto &stmt : after) {
              stmts.push_back(stmt);
            }
          }
        }
      } else if (ctrl->child->IsTask()) {
        auto task = static_cast<TaskNode *>(ctrl->child.get());
        stmts.push_back(ConvertIRStructureToStmt(task, outer_enable_epi));
      } else {
        LOG(FATAL);
      }
      Stmt body = SeqStmt::Flatten(stmts);
      // Filter out "num_stages" annotation
      Map<String, Any> filtered_annotations = ctrl->control->annotations;
      filtered_annotations.erase("num_stages");
      return For(loop_var, loop_start, loop_extent, ctrl->control->kind, body,
                 ctrl->control->thread_binding, filtered_annotations);
    }
    auto seq = static_cast<SequenceNode *>(ctrl->child.get());
    Stmt body = Evaluate(0);
    std::vector<std::vector<Stmt>> unit_stages;
    unit_stages.resize(max_stages - min_stages + 1);
    for (auto &child : seq->children) {
      auto unit = static_cast<ScheduleUnit *>(child.get());
      std::vector<Stmt> stmts;
      for (const auto &[_, before] : unit->before) {
        for (const auto &stmt : before) {
          stmts.push_back(stmt);
        }
      }
      stmts.push_back(
          ConvertIRStructureToStmt(unit->child.get(), outer_enable_epi));
      for (const auto &[_, after] : unit->after) {
        for (const auto &stmt : after) {
          stmts.push_back(stmt);
        }
      }
      unit_stages[unit->stage - min_stages].push_back(SeqStmt::Flatten(stmts));
    }
    // Set enable_pro to true only if:
    // 1. No node contains loop_break
    // 2. Loop boundaries (min and extent) are constants
    bool enable_pro = !(ctrl->child && ctrl->child->ContainsLoopBreak());

    // Check if loop boundaries are constants
    bool loop_min_is_const = tir::is_const_int(loop_start);
    bool loop_extent_is_const = tir::is_const_int(loop_extent);

    if (!loop_min_is_const || !loop_extent_is_const) {
      enable_pro = false;
    }

    bool enable_epi = enable_pro;

    // Read num_stages from loop annotation for prologue extent
    int num_stages_annotation = max_stages - min_stages;
    auto num_stages_val = ctrl->control.get()->annotations.Get("num_stages");
    if (num_stages_val.has_value()) {
      num_stages_annotation = num_stages_val.value().cast<IntImm>()->value;
    }
    int prologue_extent = 2 * num_stages_annotation;
    int epilogue_extent = max_stages - min_stages;

    std::vector<Stmt> steady;

    for (auto &child : seq->children) {
      auto unit = static_cast<ScheduleUnit *>(child.get());
      std::vector<Stmt> stmts;
      for (const auto &[_, before] : unit->before) {
        for (const auto &stmt : before) {
          stmts.push_back(stmt);
        }
      }
      stmts.push_back(
          ConvertIRStructureToStmt(unit->child.get(), outer_enable_epi));
      for (const auto &[_, after] : unit->after) {
        for (const auto &stmt : after) {
          stmts.push_back(stmt);
        }
      }
      Map<Var, PrimExpr> substitution, substitution_cond;
      substitution.Set(loop_var,
                       loop_var - loop_step * (max_stages - unit->stage));
      substitution_cond.Set(
          loop_var, Max(loop_start, Min(loop_start + loop_extent - loop_step,
                                        loop_var - loop_step * (max_stages -
                                                                unit->stage))));
      if (IsLetDeclNode(unit->child.get())) {
        Stmt stmt = SeqStmt::Flatten(stmts);
        steady.push_back(Substitute(stmt, substitution_cond));
      } else {
        PrimExpr condition =
            And(loop_var < loop_start + loop_extent, loop_var >= loop_start);
        if (unit->stage == min_stages) {
          condition = loop_var >= loop_start;
        }
        if (unit->stage == max_stages) {
          condition = loop_var < loop_start + loop_extent;
        }
        Stmt stmt = IfThenElse(condition, SeqStmt::Flatten(stmts));
        steady.push_back(Substitute(stmt, substitution));
      }
    }
    Stmt new_body = SeqStmt::Flatten(steady);
    auto new_var = loop_var.copy_with_suffix("");
    // Filter out "num_stages" annotation
    Map<String, Any> filtered_annotations = ctrl->control->annotations;
    filtered_annotations.erase("num_stages");
    Map<Var, PrimExpr> substitution;
    substitution.Set(loop_var, new_var);
    For for_op =
        For(new_var, loop_start,
            ctrl->control->extent + loop_step * (max_stages - min_stages),
            ctrl->control->kind, Substitute(new_body, substitution),
            ctrl->control->thread_binding, filtered_annotations);

    Stmt prologue = Evaluate(0);
    if (enable_pro) {
      Map<Var, PrimExpr> sub;
      For new_for = for_op;
      auto pro = loop_var.copy_with_suffix("_prologue");
      sub.Set(new_var, pro);
      new_for.CopyOnWrite()->loop_var = pro;
      new_for.CopyOnWrite()->kind = ForKind::kUnrolled;
      new_for.CopyOnWrite()->extent =
          min(prologue_extent, for_op.get()->extent);
      for_op.CopyOnWrite()->min += loop_step * prologue_extent;
      for_op.CopyOnWrite()->extent =
          max(0, for_op.get()->extent - prologue_extent);
      prologue = Substitute(new_for, sub);
      prologue = RenameLetStmtVars(prologue, "_prologue");
    }
    Stmt epilogue = Evaluate(0);
    if (enable_epi) {
      Map<Var, PrimExpr> sub;
      For new_for = for_op;
      auto epi = loop_var.copy_with_suffix("_epilogue");
      sub.Set(new_var, epi);
      new_for.CopyOnWrite()->loop_var = epi;
      new_for.CopyOnWrite()->kind = ForKind::kUnrolled;
      new_for.CopyOnWrite()->min =
          for_op.get()->min +
          loop_step * (for_op.get()->extent - epilogue_extent);
      new_for.CopyOnWrite()->extent =
          min(epilogue_extent, for_op.get()->extent);
      for_op.CopyOnWrite()->extent =
          max(0, for_op.get()->extent - epilogue_extent);
      epilogue = Substitute(new_for, sub);
      epilogue = RenameLetStmtVars(epilogue, "_epilogue");
    }
    return SeqStmt({prologue, for_op, epilogue});
  } else if (structure->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(structure);
    Stmt body = Evaluate(0);
    if (wrapper->child) {
      body = ConvertIRStructureToStmt(wrapper->child.get(), outer_enable_epi);
    }
    if (const auto *let = wrapper->wrapper.as<LetStmtNode>()) {
      return LetStmt(let->var, let->value, body);
    } else if (const auto *attr = wrapper->wrapper.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    } else {
      LOG(FATAL);
    }
  } else if (structure->IsIf()) {
    auto if_node = static_cast<const IfNode *>(structure);
    Stmt then_stmt =
        ConvertIRStructureToStmt(if_node->then_child.get(), outer_enable_epi);
    Optional<Stmt> else_stmt;
    if (if_node->else_child) {
      else_stmt =
          ConvertIRStructureToStmt(if_node->else_child.get(), outer_enable_epi);
    }
    return IfThenElse(if_node->condition, then_stmt, else_stmt);
  }

  LOG(FATAL)
      << "Failed to convert IRStructure to Stmt, returning empty statement";
  return Evaluate(0);
}

// Apply warpgroup partition to entire IRStructure (top-level IfThenElse)
Stmt ApplyWarpgroupPartitionToIRStructure(
    IRStructure *root, IterVar thread_var, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, const bool outer_enable_epi,
    const std::vector<PrimExpr> &thread_count,
    const WarpSpecializeConfig &config, Buffer neutral_sync_shared_barrier,
    std::vector<Buffer> &duplicated_fragment_buffers) {
  if (!root)
    return Evaluate(0);

  if (root->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(root);
    Stmt body = Evaluate(0);
    if (wrapper->child) {
      body = ApplyWarpgroupPartitionToIRStructure(
          wrapper->child.get(), thread_var, barrier_buffers, barrier_map,
          outer_enable_epi, thread_count, config, neutral_sync_shared_barrier,
          duplicated_fragment_buffers);
    }
    if (const auto *let = wrapper->wrapper.as<LetStmtNode>()) {
      return LetStmt(let->var, let->value, body);
    } else if (const auto *attr = wrapper->wrapper.as<AttrStmtNode>()) {
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    } else {
      LOG(FATAL);
      return Evaluate(0);
    }
  }

  size_t num_wgs = thread_count.size();

  auto has_actual_statements = [](IRStructure *node) -> bool {
    if (!node)
      return false;
    std::vector<TaskNodeWithContext> tasks;
    CollectAllTaskNodesWithContext(node, tasks);
    for (auto &task : tasks) {
      if (!task.task->stmts.empty()) {
        return true;
      }
    }
    return false;
  };

  std::function<std::shared_ptr<IRStructure>(IRStructure *, SchedulePhase)>
      clone_phase_filter;
  clone_phase_filter =
      [&clone_phase_filter](
          IRStructure *node,
          SchedulePhase phase) -> std::shared_ptr<IRStructure> {
    if (!node)
      return nullptr;

    if (node->IsTask()) {
      auto task = static_cast<TaskNode *>(node);
      if (task->GetSchedulePhase() == phase) {
        return task->Clone();
      } else {
        auto new_task = std::make_shared<TaskNode>();
        return new_task;
      }
    } else if (node->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(node);
      auto new_seq = std::make_shared<SequenceNode>();
      for (const auto &child : seq->children) {
        if (child) {
          auto schedule_unit = static_cast<ScheduleUnit *>(child.get());
          auto new_node = clone_phase_filter(schedule_unit->child.get(), phase);
          if (new_node) {
            auto new_unit = std::make_shared<ScheduleUnit>();
            new_unit->child = std::move(new_node);
            new_seq->children.push_back(std::move(new_unit));
          }
        }
      }
      return new_seq;
    } else if (node->IsWrapper()) {
      auto wrapper = static_cast<WrapperNode *>(node);
      auto new_wrapper = std::make_shared<WrapperNode>();
      new_wrapper->child = clone_phase_filter(wrapper->child.get(), phase);
      if (new_wrapper->child) {
        return new_wrapper;
      }
      return nullptr;
    } else if (node->IsControl()) {
      return nullptr;
    } else if (node->IsIf()) {
      return nullptr;
    }
    LOG(FATAL);
    return nullptr;
  };

  auto wg_pro_neutral_structure =
      clone_phase_filter(root, SchedulePhase::kPrologue);
  auto wg_epi_neutral_structure =
      clone_phase_filter(root, SchedulePhase::kEpilogue);
  // wg_children[wg_id][child_index] = filtered IRStructure (nullptr if absent)
  std::vector<std::vector<std::shared_ptr<IRStructure>>> wg_children(num_wgs);
  std::vector<std::shared_ptr<IRStructure>> wg_structures(num_wgs);

  std::vector<Buffer> broadcast_fragments =
      CollectBroadcastFragmentBuffers(root);
  std::vector<Map<Buffer, Buffer>> per_wg_buffer_remap(num_wgs);
  if (!broadcast_fragments.empty()) {
    for (size_t i = 1; i < num_wgs; ++i) {
      std::string suffix = "_wg" + std::to_string(i);
      for (const auto &buf : broadcast_fragments) {
        Buffer new_buf = DuplicateFragmentBuffer(buf, suffix);
        per_wg_buffer_remap[i].Set(buf, new_buf);
        duplicated_fragment_buffers.push_back(new_buf);
      }
    }
  }

  if (root->IsSequence()) {
    auto root_seq = static_cast<SequenceNode *>(root);
    for (size_t i = 0; i < num_wgs; ++i) {
      Map<Var, PrimExpr> var_remap;
      wg_children[i] = CloneIRStructureChildrenWithWarpgroupFilter(
          root_seq, i, var_remap, per_wg_buffer_remap[i]);
    }
    for (size_t i = 0; i < num_wgs; ++i) {
      // Rebuild from wg_children: wrap non-null children into a SequenceNode
      auto rebuilt_seq = std::make_shared<SequenceNode>();
      for (const auto &child : wg_children[i]) {
        if (child)
          rebuilt_seq->children.push_back(child);
      }
      wg_structures[i] = rebuilt_seq->children.empty() ? nullptr : rebuilt_seq;
    }
  } else {
    // Fallback for non-SequenceNode root: clone entire root per warpgroup.
    for (size_t i = 0; i < num_wgs; ++i) {
      Map<Var, PrimExpr> var_remap;
      wg_structures[i] = CloneIRStructureWithWarpgroupFilter(
          root, i, var_remap, per_wg_buffer_remap[i]);
    }
  }

  std::vector<PrimExpr> wg_conditions(num_wgs);
  wg_conditions[0] = thread_count[0];
  for (size_t i = 1; i < num_wgs; ++i) {
    wg_conditions[i] = wg_conditions[i - 1] + thread_count[i];
  }
  for (auto &cond : wg_conditions) {
    cond = thread_var->var < cond;
  }

  bool wg_pro_neutral_has_stmts =
      has_actual_statements(wg_pro_neutral_structure.get());
  bool wg_epi_neutral_has_stmts =
      has_actual_statements(wg_epi_neutral_structure.get());

  Stmt pro_neutral_body =
      wg_pro_neutral_has_stmts
          ? ConvertIRStructureToStmt(wg_pro_neutral_structure.get(),
                                     outer_enable_epi)
          : Evaluate(0);
  Stmt epi_neutral_body =
      wg_epi_neutral_has_stmts
          ? ConvertIRStructureToStmt(wg_epi_neutral_structure.get(),
                                     outer_enable_epi)
          : Evaluate(0);

  // Helper: build a single IfThenElse (with wg nesting) from per-wg Stmts.
  auto MakeWarpgroupIf =
      [&wg_conditions](const std::vector<Stmt> &wg_stmts) -> Stmt {
    Stmt if_then_else = Evaluate(0);
    for (size_t i = wg_stmts.size(); i-- > 0;) {
      if_then_else = IfThenElse(wg_conditions[i], wg_stmts[i], if_then_else);
    }
    return if_then_else;
  };

  // Check for SIMT copy in wg1 (needed for set_max_nreg decision).
  bool has_simt_copy = false;
  if (num_wgs == 2 && wg_structures[1]) {
    Stmt full_wg1 =
        ConvertIRStructureToStmt(wg_structures[1].get(), outer_enable_epi);
    has_simt_copy = SimtCopyDetector::Detect(full_wg1);
  }

  // Check whether any inner pass already decided register allocation.
  bool has_inner_nreg_decision = false;
  if (num_wgs == 2 && config.enable_set_max_nreg) {
    for (size_t i = 0; i < num_wgs; ++i) {
      if (!wg_structures[i]) {
        continue;
      }
      Stmt full_wg =
          ConvertIRStructureToStmt(wg_structures[i].get(), outer_enable_epi);
      if (InnerNRegDecisionDetector::Detect(full_wg)) {
        has_inner_nreg_decision = true;
        break;
      }
    }
  }

  // --- Per-child construction ---
  // Walk root SequenceNode's children.  LetDecl children accumulate bindings;
  // non-LetDecl children produce IfThenElse blocks wrapped with accumulated
  // LetDecl scopes per warp group.

  Stmt if_then_else;
  if (root->IsSequence()) {
    auto root_seq = static_cast<SequenceNode *>(root);
    size_t num_children = root_seq->children.size();

    // per-wg accumulated LetDecl {var, value} from earlier children
    std::vector<std::vector<std::pair<Var, PrimExpr>>> wg_accumulated_lets(
        num_wgs);

    std::vector<Stmt> segmented_stmts;
    bool first_non_let = true;

    for (size_t ci = 0; ci < num_children; ++ci) {
      auto unit = static_cast<ScheduleUnit *>(root_seq->children[ci].get());
      bool is_let_decl = IsLetDeclNode(unit->child.get());

      if (is_let_decl) {
        // Extract LetDecl {var, value} from each wg's filtered result
        for (size_t i = 0; i < num_wgs; ++i) {
          if (wg_children[i][ci]) {
            // wg_children[i][ci] is a ScheduleUnit wrapping a TaskNode
            IRStructure *inner = wg_children[i][ci].get();
            TaskNode *task = nullptr;
            if (inner->IsScheduleUnit()) {
              task = static_cast<TaskNode *>(
                  static_cast<ScheduleUnit *>(inner)->child.get());
            } else if (inner->IsTask()) {
              task = static_cast<TaskNode *>(inner);
            }
            if (task && !task->stmts.empty()) {
              const auto *let = task->stmts[0].as<LetStmtNode>();
              if (let) {
                wg_accumulated_lets[i].push_back({let->var, let->value});
              }
            }
          }
        }
        continue; // LetDecl children don't produce IfThenElse
      }

      // Build per-wg Stmt for this child, wrapped with accumulated LetDecls
      std::vector<Stmt> wg_stmts(num_wgs);
      bool all_empty = true;
      for (size_t i = 0; i < num_wgs; ++i) {
        if (wg_children[i][ci]) {
          auto tmp_seq = std::make_shared<SequenceNode>();
          tmp_seq->children.push_back(wg_children[i][ci]);
          wg_stmts[i] =
              ConvertIRStructureToStmt(tmp_seq.get(), outer_enable_epi);
        } else {
          wg_stmts[i] = Evaluate(0);
        }
        if (!IsEvaluateZero(wg_stmts[i])) {
          all_empty = false;
        }
        // Wrap with accumulated LetDecl bindings (innermost first)
        for (int j = static_cast<int>(wg_accumulated_lets[i].size()) - 1;
             j >= 0; --j) {
          wg_stmts[i] = LetStmt(wg_accumulated_lets[i][j].first,
                                wg_accumulated_lets[i][j].second, wg_stmts[i]);
        }
      }

      // Skip segments where all warpgroups produce empty statements
      if (all_empty)
        continue;

      auto PeelLetsToInner = [](const Stmt &s) -> Stmt {
        const Stmt *cur = &s;
        while (const auto *let = cur->as<LetStmtNode>()) {
          cur = &let->body;
        }
        return *cur;
      };
      bool is_shared_attr_segment = true;
      Stmt shared_attr_stmt;
      for (size_t i = 0; i < num_wgs; ++i) {
        if (IsEvaluateZero(wg_stmts[i])) {
          continue;
        }
        Stmt inner = PeelLetsToInner(wg_stmts[i]);
        const auto *attr = inner.as<AttrStmtNode>();
        if (!attr || !IsEvaluateZero(attr->body)) {
          is_shared_attr_segment = false;
          break;
        }
        if (!shared_attr_stmt.defined()) {
          shared_attr_stmt = wg_stmts[i];
        }
      }

      // Insert liveness boundary before each non-empty non-LetDecl child
      segmented_stmts.push_back(AttrStmt(
          Integer(0), attr::kAutoScheduleSharedMemoryBoundary, 0, Evaluate(0)));

      if (is_shared_attr_segment && shared_attr_stmt.defined()) {
        segmented_stmts.push_back(shared_attr_stmt);
        continue;
      }

      // Prepend set_max_nreg only to the first non-LetDecl child
      if (first_non_let && !has_simt_copy && !has_inner_nreg_decision &&
          num_wgs == 2 && config.enable_set_max_nreg) {
        for (size_t i = 0; i < num_wgs; ++i) {
          wg_stmts[i] =
              SeqStmt({Evaluate(Call(DataType::Handle(), tl::set_max_nreg(),
                                     {i == 0 ? config.consumer_max_nreg
                                             : config.producer_max_nreg,
                                      static_cast<int>(!i)})),
                       wg_stmts[i]});
        }
      }
      first_non_let = false;

      segmented_stmts.push_back(MakeWarpgroupIf(wg_stmts));
    }
    segmented_stmts.push_back(AttrStmt(
        Integer(0), attr::kAutoScheduleSharedMemoryBoundary, 0, Evaluate(0)));
    if_then_else = SeqStmt::Flatten(segmented_stmts);
  } else {
    // Fallback for non-SequenceNode root: no boundary insertion, simple
    // partition
    std::vector<Stmt> wg_stmts(num_wgs);
    for (size_t i = 0; i < num_wgs; ++i) {
      if (wg_structures[i]) {
        wg_stmts[i] =
            ConvertIRStructureToStmt(wg_structures[i].get(), outer_enable_epi);
      } else {
        wg_stmts[i] = Evaluate(0);
      }
    }
    if (!has_simt_copy && !has_inner_nreg_decision && num_wgs == 2 &&
        config.enable_set_max_nreg) {
      for (size_t i = 0; i < num_wgs; ++i) {
        wg_stmts[i] =
            SeqStmt({Evaluate(Call(DataType::Handle(), tl::set_max_nreg(),
                                   {i == 0 ? config.consumer_max_nreg
                                           : config.producer_max_nreg,
                                    static_cast<int>(!i)})),
                     wg_stmts[i]});
      }
    }
    if_then_else = MakeWarpgroupIf(wg_stmts);
  }

  PrimExpr updated_thread_extent = std::accumulate(
      thread_count.begin() + 1, thread_count.end(), thread_count[0]);

  Stmt pro_and_warpgroup_stmt;
  if (wg_pro_neutral_has_stmts && !IsEvaluateZero(pro_neutral_body)) {
    pro_and_warpgroup_stmt = InsertBarriersForNeutralSync(
        pro_neutral_body, if_then_else, barrier_buffers, barrier_map,
        updated_thread_extent, neutral_sync_shared_barrier);
  } else {
    pro_and_warpgroup_stmt = if_then_else;
  }

  bool need_shared_barrier_for_epi = false;
  bool need_tmem_barrier_for_epi = false;
  if (wg_epi_neutral_structure) {
    for (const auto &warpgroup_structure : wg_structures) {
      need_shared_barrier_for_epi =
          need_shared_barrier_for_epi ||
          HasSharedWriteReadDependency(warpgroup_structure.get(),
                                       wg_epi_neutral_structure.get());
      need_tmem_barrier_for_epi =
          need_tmem_barrier_for_epi ||
          HasTmemWriteReadDependency(warpgroup_structure.get(),
                                     wg_epi_neutral_structure.get());
    }
  }

  Stmt combined_stmt;
  if (!IsEvaluateZero(pro_and_warpgroup_stmt) &&
      !IsEvaluateZero(epi_neutral_body)) {
    // Both have statements: insert barriers for warpgroup-to-epi_neutral
    // synchronization
    // TODO: tensor core may not only in wg0?
    combined_stmt = InsertBarriersForNeutralSyncWithDependency(
        pro_and_warpgroup_stmt, epi_neutral_body, barrier_buffers, barrier_map,
        updated_thread_extent, need_shared_barrier_for_epi,
        need_tmem_barrier_for_epi, Buffer(), thread_var->var, 0,
        thread_count[0]);
  } else if (!IsEvaluateZero(epi_neutral_body)) {
    combined_stmt = epi_neutral_body;
  } else {
    combined_stmt = pro_and_warpgroup_stmt;
  }

  return SeqStmt({AttrStmt(Integer(0), attr::kAutoScheduleSharedMemoryBoundary,
                           0, Evaluate(0)),
                  combined_stmt});
}

} // namespace tl
} // namespace tvm
