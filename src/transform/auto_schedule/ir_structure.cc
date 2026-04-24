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
 * \file ir_structure.cc
 * \brief IR structure analysis for TileLang
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../op/builtin.h"
#include "../common/attr.h"
#include "../common/collector.h"
#include "./ir_structure.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

// SequenceNode member function implementations
bool SequenceNode::UsesCUDACore() const {
  for (const auto &child : children) {
    if (child && child->UsesCUDACore())
      return true;
  }
  return false;
}

bool SequenceNode::UsesTMACore() const {
  for (const auto &child : children) {
    if (child && child->UsesTMACore())
      return true;
  }
  return false;
}

bool SequenceNode::UsesTensorCore() const {
  for (const auto &child : children) {
    if (child && child->UsesTensorCore())
      return true;
  }
  return false;
}

bool SequenceNode::HasWGMMA() const {
  for (const auto &child : children) {
    if (child && child->HasWGMMA())
      return true;
  }
  return false;
}

bool SequenceNode::HasTCGEN05() const {
  for (const auto &child : children) {
    if (child && child->HasTCGEN05())
      return true;
  }
  return false;
}

std::vector<BufferRegion> SequenceNode::GetReadRegions() const {
  std::vector<BufferRegion> all_read_regions;
  for (const auto &child : children) {
    if (child) {
      auto child_read_regions = child->GetReadRegions();
      all_read_regions.insert(all_read_regions.end(),
                              child_read_regions.begin(),
                              child_read_regions.end());
    }
  }
  // Remove duplicates (by buffer and region equality)
  // This is a simplified deduplication - in practice might need more
  // sophisticated logic
  std::vector<BufferRegion> deduplicated;
  for (const auto &region : all_read_regions) {
    bool found = false;
    for (const auto &existing : deduplicated) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        found = true;
        break;
      }
    }
    if (!found)
      deduplicated.push_back(region);
  }
  return deduplicated;
}

std::vector<Var> SequenceNode::GetReadVars() const {
  std::vector<Var> all_vars;
  for (const auto &child : children) {
    if (child) {
      auto child_read_vars = child->GetReadVars();
      all_vars.insert(all_vars.end(), child_read_vars.begin(),
                      child_read_vars.end());
    }
  }
  return all_vars;
}

std::vector<BufferRegion> SequenceNode::GetWriteRegions() const {
  std::vector<BufferRegion> all_write_regions;
  for (const auto &child : children) {
    if (child) {
      auto child_write_regions = child->GetWriteRegions();
      all_write_regions.insert(all_write_regions.end(),
                               child_write_regions.begin(),
                               child_write_regions.end());
    }
  }
  // Remove duplicates (by buffer and region equality)
  std::vector<BufferRegion> deduplicated;
  for (const auto &region : all_write_regions) {
    bool found = false;
    for (const auto &existing : deduplicated) {
      if (existing->buffer.same_as(region->buffer) &&
          RegionsEqual(existing->region, region->region)) {
        found = true;
        break;
      }
    }
    if (!found)
      deduplicated.push_back(region);
  }
  return deduplicated;
}

std::vector<Var> SequenceNode::GetWriteVars() const {
  std::vector<Var> all_vars;
  for (const auto &child : children) {
    if (child) {
      auto child_write_vars = child->GetWriteVars();
      all_vars.insert(all_vars.end(), child_write_vars.begin(),
                      child_write_vars.end());
    }
  }
  return all_vars;
}

int64_t SequenceNode::GetLatency() const { return latency_; }

int64_t SequenceNode::GetII() const { return ii_; }

void SequenceNode::SetUsesCUDACore(bool value) {
  for (auto &child : children) {
    if (child)
      child->SetUsesCUDACore(value);
  }
}

void SequenceNode::SetUsesTMACore(bool value) {
  for (auto &child : children) {
    if (child)
      child->SetUsesTMACore(value);
  }
}

void SequenceNode::SetUsesTensorCore(bool value) {
  for (auto &child : children) {
    if (child)
      child->SetUsesTensorCore(value);
  }
}

void SequenceNode::SetReadRegions(const std::vector<BufferRegion> &regions) {
  // Not clear what this means for SequenceNode - maybe set on first child?
  if (!children.empty() && children[0]) {
    children[0]->SetReadRegions(regions);
  }
}

void SequenceNode::SetWriteRegions(const std::vector<BufferRegion> &regions) {
  if (!children.empty() && children[0]) {
    children[0]->SetWriteRegions(regions);
  }
}

void SequenceNode::SetLatency(int64_t latency) { latency_ = latency; }

void SequenceNode::SetII(int64_t ii) { ii_ = ii; }

std::shared_ptr<IRStructure> SequenceNode::Clone() const {
  auto new_seq = std::make_shared<SequenceNode>();
  new_seq->children.reserve(children.size());
  for (const auto &child : children) {
    if (child) {
      new_seq->children.push_back(child->Clone());
    } else {
      new_seq->children.push_back(nullptr);
    }
  }
  // Copy latency and II
  new_seq->SetLatency(GetLatency());
  new_seq->SetII(GetII());
  return new_seq;
}

std::shared_ptr<IRStructure> TaskNode::Clone() const {
  auto new_task = std::make_shared<TaskNode>();
  // Copy statements
  new_task->stmts = stmts;
  // Copy resource usage flags
  new_task->SetUsesCUDACore(UsesCUDACore());
  new_task->SetUsesTMACore(UsesTMACore());
  new_task->SetUsesTensorCore(UsesTensorCore());
  // Copy memory access regions
  new_task->SetReadRegions(GetReadRegions());
  new_task->SetWriteRegions(GetWriteRegions());
  // Copy latency and II
  new_task->SetLatency(GetLatency());
  new_task->SetII(GetII());
  // Copy start time
  new_task->SetStartTime(GetStartTime());
  // Copy warpgroup id
  new_task->SetWarpgroupId(GetWarpgroupId());
  // Copy scheduling phase
  new_task->SetSchedulePhase(GetSchedulePhase());
  // Copy loop_break cache
  new_task->contains_loop_break_cache_ = contains_loop_break_cache_;
  return new_task;
}

void TaskNode::CollectBufferAccessInfo(
    int num_wgs, SchedulePhase phase,
    std::set<BufferAccessInfo> &result) const {
  int wg_id = GetWarpgroupId();
  if (GetSchedulePhase() != phase) {
    return;
  }

  // Helper: emit buffer access for a single region.
  auto emit_access = [&](const BufferRegion &region, bool is_write) {
    if (wg_id >= 0) {
      // Normal assigned warpgroup
      result.emplace(region->buffer, is_write, wg_id, this);
    } else if (IsWarpgroupBroadcast(wg_id)) {
      // Broadcast: skip register memory (each wg has its own copy), removed for barrier analysis.
      // if (IsRegisterRegion(region)) {
      //   return;
      // }
      // Shared/global memory is shared across wgs — emit for all
      for (int i = 0; i < num_wgs; ++i) {
        result.emplace(region->buffer, is_write, i, this);
      }
    } else {
      // Unassigned (kWarpgroupUnassigned): expand to all wgs (legacy behavior)
      for (int i = 0; i < num_wgs; ++i) {
        result.emplace(region->buffer, is_write, i, this);
      }
    }
  };

  // Collect write buffers
  for (const auto &region : GetWriteRegions()) {
    emit_access(region, true);
  }
  // Collect read buffers
  for (const auto &region : GetReadRegions()) {
    emit_access(region, false);
  }
}

bool TaskNode::ContainsLoopBreak() const {
  // Return cached result if available
  if (contains_loop_break_cache_.has_value()) {
    return contains_loop_break_cache_.value();
  }

  // Check if any statement in this task contains a loop_break call
  bool found_loop_break = false;
  for (const auto &stmt : stmts) {
    // Helper function to check if a statement contains loop_break
    auto contains_loop_break = [](const Stmt &stmt) -> bool {
      bool found = false;
      PostOrderVisit(stmt, [&found](const ObjectRef &node) {
        if (found)
          return;
        if (const auto *call = node.as<CallNode>()) {
          if (call->op.same_as(tl::loop_break())) {
            found = true;
          }
        }
      });
      return found;
    };

    if (contains_loop_break(stmt)) {
      found_loop_break = true;
      break;
    }
  }

  contains_loop_break_cache_ = found_loop_break;
  return found_loop_break;
}

std::shared_ptr<IRStructure> ControlNode::Clone() const {
  auto new_ctrl = std::make_shared<ControlNode>();
  // Copy For control (For is a TVM object with reference counting)
  new_ctrl->control = control;
  // Clone child if exists
  if (child) {
    new_ctrl->child = child->Clone();
  }
  // Copy latency and II
  new_ctrl->SetLatency(GetLatency());
  new_ctrl->SetII(GetII());
  new_ctrl->SetPromote(hasPromote());
  return new_ctrl;
}

std::shared_ptr<IRStructure> WrapperNode::Clone() const {
  auto new_wrapper = std::make_shared<WrapperNode>();
  // Copy var and value (TVM objects with reference counting)
  new_wrapper->wrapper = wrapper;
  // Clone child if exists
  if (child) {
    new_wrapper->child = child->Clone();
  }
  // Copy latency and II
  new_wrapper->SetLatency(GetLatency());
  new_wrapper->SetII(GetII());
  return new_wrapper;
}

std::shared_ptr<IRStructure> ScheduleUnit::Clone() const {
  auto new_unit = std::make_shared<ScheduleUnit>();
  // Copy var and value (TVM objects with reference counting)
  new_unit->stage = stage;
  // Clone child if exists
  if (child) {
    new_unit->child = child->Clone();
  }
  // Copy latency and II
  new_unit->SetLatency(GetLatency());
  new_unit->SetII(GetII());
  return new_unit;
}

std::shared_ptr<IRStructure> IfNode::Clone() const {
  auto new_if = std::make_shared<IfNode>();
  new_if->condition = condition;
  if (then_child) {
    new_if->then_child = then_child->Clone();
  }
  if (else_child) {
    new_if->else_child = else_child->Clone();
  }
  if (task) {
    new_if->task = std::static_pointer_cast<TaskNode>(task->Clone());
  }
  new_if->SetLatency(GetLatency());
  new_if->SetII(GetII());
  new_if->SetStartTime(GetStartTime());
  return new_if;
}

void ControlNode::CollectBufferAccessInfo(
    int num_wgs, SchedulePhase phase,
    std::set<BufferAccessInfo> &result) const {
  if (task) {
    task->CollectBufferAccessInfo(num_wgs, phase, result);
  }
  if (child) {
    child->CollectBufferAccessInfo(num_wgs, phase, result);
  }
}

void WrapperNode::CollectBufferAccessInfo(
    int num_wgs, SchedulePhase phase,
    std::set<BufferAccessInfo> &result) const {
  if (task) {
    task->CollectBufferAccessInfo(num_wgs, phase, result);
  }
  if (child) {
    child->CollectBufferAccessInfo(num_wgs, phase, result);
  }
}

void ScheduleUnit::CollectBufferAccessInfo(
    int num_wgs, SchedulePhase phase,
    std::set<BufferAccessInfo> &result) const {
  if (child) {
    child->CollectBufferAccessInfo(num_wgs, phase, result);
  }
}

void SequenceNode::CollectBufferAccessInfo(
    int num_wgs, SchedulePhase phase,
    std::set<BufferAccessInfo> &result) const {
  for (const auto &child : children) {
    if (child) {
      child->CollectBufferAccessInfo(num_wgs, phase, result);
    }
  }
}

void IfNode::CollectBufferAccessInfo(int num_wgs, SchedulePhase phase,
                                     std::set<BufferAccessInfo> &result) const {
  if (task) {
    task->CollectBufferAccessInfo(num_wgs, phase, result);
  }
  if (then_child) {
    then_child->CollectBufferAccessInfo(num_wgs, phase, result);
  }
  if (else_child) {
    else_child->CollectBufferAccessInfo(num_wgs, phase, result);
  }
}

// Helper function to collect all TaskNodes with context information
void CollectAllTaskNodesWithContext(IRStructure *node,
                                    std::vector<TaskNodeWithContext> &all_tasks,
                                    ControlNode *current_control_node) {
  if (!node)
    return;

  if (node->IsTask()) {
    auto task = static_cast<TaskNode *>(node);
    TaskNodeWithContext task_ctx;
    task_ctx.task = task;
    task_ctx.control_node = current_control_node;

    // Calculate tripcount if inside a loop
    if (current_control_node) {
      const ForNode *for_node = current_control_node->control.get();
      PrimExpr loop_extent = for_node->extent;
      PrimExpr loop_step = for_node->step.has_value()
                               ? for_node->step.value()
                               : IntImm(DataType::Int(32), 1);

      // Try to convert loop_extent and loop_step to int64_t
      if (const int64_t *extent_ptr = as_const_int(loop_extent)) {
        if (const int64_t *step_ptr = as_const_int(loop_step)) {
          // Calculate ceil(extent / step)
          int64_t extent = *extent_ptr;
          int64_t step = *step_ptr;
          if (step > 0) {
            // ceil(extent / step) = (extent + step - 1) / step
            task_ctx.tripcount = (extent + step - 1) / step;
          } else {
            // Invalid step, use extent as fallback
            task_ctx.tripcount = extent;
          }
        } else {
          // Step is not constant, use 100 as default
          task_ctx.tripcount = 100;
        }
      } else {
        // Extent is not constant, use 100 as default (as requested)
        task_ctx.tripcount = 100;
      }
    } else {
      task_ctx.tripcount = 1; // Not inside a loop
    }

    all_tasks.push_back(task_ctx);
  } else if (node->IsSequence()) {
    auto seq = static_cast<const SequenceNode *>(node);
    for (const auto &child : seq->children) {
      CollectAllTaskNodesWithContext(child.get(), all_tasks,
                                     current_control_node);
    }
  } else if (node->IsControl()) {
    auto control = static_cast<const ControlNode *>(node);
    // When entering a control node, update the current control context
    CollectAllTaskNodesWithContext(control->child.get(), all_tasks,
                                   const_cast<ControlNode *>(control));
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<const WrapperNode *>(node);
    // Wrapper nodes don't change control context, just recurse into child
    CollectAllTaskNodesWithContext(wrapper->child.get(), all_tasks,
                                   current_control_node);
  } else if (node->IsScheduleUnit()) {
    auto promote = static_cast<const ScheduleUnit *>(node);
    // Promote nodes don't change control context, just recurse into child
    CollectAllTaskNodesWithContext(promote->child.get(), all_tasks,
                                   current_control_node);
  } else if (node->IsIf()) {
    auto if_node = static_cast<const IfNode *>(node);
    // Recurse into both branches
    CollectAllTaskNodesWithContext(if_node->then_child.get(), all_tasks,
                                   current_control_node);
    if (if_node->else_child) {
      CollectAllTaskNodesWithContext(if_node->else_child.get(), all_tasks,
                                     current_control_node);
    }
  } else {
    LOG(FATAL);
  }
}

// ============================================================================
// CollectFirstAccessTasks / CollectLastAccessTasks implementations
//
// These methods return the set of TaskNode pointers that could possibly be the
// first (or last) to perform a specific buffer access (buffer, is_write, wg_id)
// within the IR subtree.  The bool return value indicates whether the subtree
// is *guaranteed* to contain at least one matching access (must_have).
// ============================================================================

static const IfNode *TryGetIfNode(const IRStructure *node) {
  if (!node)
    return nullptr;
  if (node->IsIf())
    return static_cast<const IfNode *>(node);
  if (node->IsScheduleUnit()) {
    auto *unit = static_cast<const ScheduleUnit *>(node);
    if (unit->child && unit->child->IsIf())
      return static_cast<const IfNode *>(unit->child.get());
  }
  return nullptr;
}

static bool
SequenceCollectFirstAccessTasks(const std::vector<const IRStructure *> &nodes,
                                const Buffer &buffer, bool is_write, int wg_id,
                                SchedulePhase phase,
                                std::set<const TaskNode *> &result) {
  int n = static_cast<int>(nodes.size());

  // Find the first node that contains loop_break.
  int break_idx = -1;
  for (int j = 0; j < n; ++j) {
    if (nodes[j]->ContainsLoopBreak()) {
      break_idx = j;
      break;
    }
  }
  if (break_idx >= 0) {
    if (break_idx > 0) {
      std::vector<const IRStructure *> before(nodes.begin(),
                                              nodes.begin() + break_idx);
      if (SequenceCollectFirstAccessTasks(before, buffer, is_write, wg_id,
                                          phase, result))
        return true;
    }
    if (nodes[break_idx]->CollectFirstAccessTasks(buffer, is_write, wg_id,
                                                  phase, result))
      return true;
    if (break_idx + 1 < n) {
      std::vector<const IRStructure *> after(nodes.begin() + break_idx + 1,
                                             nodes.end());
      SequenceCollectFirstAccessTasks(after, buffer, is_write, wg_id, phase,
                                      result);
    }
    return false;
  }

  // No loop_break
  int i = 0;
  while (i < n) {
    const IfNode *if_node = TryGetIfNode(nodes[i]);
    if (if_node) {
      int group_start = i;
      while (i + 1 < n) {
        const IfNode *next_if = TryGetIfNode(nodes[i + 1]);
        if (!next_if)
          break;
        if (!StructuralEqual()(if_node->condition, next_if->condition))
          break;
        ++i;
      }
      int group_end = i;
      ++i;
      if (if_node->task) {
        if (if_node->task->CollectFirstAccessTasks(buffer, is_write, wg_id,
                                                   phase, result))
          return true;
      }
      std::vector<const IRStructure *> then_children, else_children;
      for (int j = group_start; j <= group_end; ++j) {
        const IfNode *cur_if = TryGetIfNode(nodes[j]);
        ICHECK(cur_if);
        if (cur_if->then_child)
          then_children.push_back(cur_if->then_child.get());
        if (cur_if->else_child)
          else_children.push_back(cur_if->else_child.get());
      }
      bool then_must = false, else_must = false;
      if (!then_children.empty())
        then_must = SequenceCollectFirstAccessTasks(
            then_children, buffer, is_write, wg_id, phase, result);
      if (!else_children.empty())
        else_must = SequenceCollectFirstAccessTasks(
            else_children, buffer, is_write, wg_id, phase, result);
      if (then_must && else_must)
        return true;
    } else {
      if (nodes[i]->CollectFirstAccessTasks(buffer, is_write, wg_id, phase,
                                            result))
        return true;
      ++i;
    }
  }
  return false;
}

static bool
SequenceCollectLastAccessTasks(const std::vector<const IRStructure *> &nodes,
                               const Buffer &buffer, bool is_write, int wg_id,
                               SchedulePhase phase,
                               std::set<const TaskNode *> &result) {
  int n = static_cast<int>(nodes.size());

  // Find the last node that contains loop_break.
  int break_idx = -1;
  for (int j = n - 1; j >= 0; --j) {
    if (nodes[j]->ContainsLoopBreak()) {
      break_idx = j;
      break;
    }
  }
  if (break_idx >= 0) {
    if (break_idx + 1 < n) {
      std::vector<const IRStructure *> after(nodes.begin() + break_idx + 1,
                                             nodes.end());
      SequenceCollectLastAccessTasks(after, buffer, is_write, wg_id, phase,
                                     result);
    }
    nodes[break_idx]->CollectLastAccessTasks(buffer, is_write, wg_id, phase,
                                             result);
    if (break_idx > 0) {
      std::vector<const IRStructure *> before(nodes.begin(),
                                              nodes.begin() + break_idx);
      SequenceCollectLastAccessTasks(before, buffer, is_write, wg_id, phase,
                                     result);
    }
    return false;
  }

  // No loop_break
  int i = n - 1;
  while (i >= 0) {
    const IfNode *if_node = TryGetIfNode(nodes[i]);
    if (if_node) {
      int group_end = i;
      while (i - 1 >= 0) {
        const IfNode *prev_if = TryGetIfNode(nodes[i - 1]);
        if (!prev_if)
          break;
        if (!StructuralEqual()(if_node->condition, prev_if->condition))
          break;
        --i;
      }
      int group_start = i;
      --i;
      std::vector<const IRStructure *> then_children, else_children;
      for (int j = group_start; j <= group_end; ++j) {
        const IfNode *cur_if = TryGetIfNode(nodes[j]);
        ICHECK(cur_if);
        if (cur_if->then_child)
          then_children.push_back(cur_if->then_child.get());
        if (cur_if->else_child)
          else_children.push_back(cur_if->else_child.get());
      }
      bool then_must = false, else_must = false;
      if (!then_children.empty())
        then_must = SequenceCollectLastAccessTasks(
            then_children, buffer, is_write, wg_id, phase, result);
      if (!else_children.empty())
        else_must = SequenceCollectLastAccessTasks(
            else_children, buffer, is_write, wg_id, phase, result);
      if (then_must && else_must)
        return true;
      const IfNode *last_if = TryGetIfNode(nodes[group_end]);
      ICHECK(last_if);
      if (last_if->task) {
        if (last_if->task->CollectLastAccessTasks(buffer, is_write, wg_id,
                                                  phase, result))
          return true;
      }
    } else {
      if (nodes[i]->CollectLastAccessTasks(buffer, is_write, wg_id, phase,
                                           result))
        return true;
      --i;
    }
  }
  return false;
}

static bool TaskMatchesAccess(const TaskNode *task, const Buffer &buffer,
                              bool is_write, int wg_id, SchedulePhase phase) {
  if (task->GetSchedulePhase() != phase)
    return false;
  int task_wg = task->GetWarpgroupId();
  const auto &regions =
      is_write ? task->GetWriteRegions() : task->GetReadRegions();
  for (const auto &region : regions) {
    if (region->buffer == buffer &&
        (task_wg == wg_id || IsWarpgroupBroadcast(task_wg)))
      return true;
  }
  return false;
}

bool TaskNode::CollectFirstAccessTasks(
    const Buffer &buffer, bool is_write, int wg_id, SchedulePhase phase,
    std::set<const TaskNode *> &result) const {
  if (TaskMatchesAccess(this, buffer, is_write, wg_id, phase)) {
    result.insert(this);
    return true;
  }
  return false;
}

bool TaskNode::CollectLastAccessTasks(
    const Buffer &buffer, bool is_write, int wg_id, SchedulePhase phase,
    std::set<const TaskNode *> &result) const {
  if (TaskMatchesAccess(this, buffer, is_write, wg_id, phase)) {
    result.insert(this);
    return true;
  }
  return false;
}

bool ScheduleUnit::CollectFirstAccessTasks(
    const Buffer &buffer, bool is_write, int wg_id, SchedulePhase phase,
    std::set<const TaskNode *> &result) const {
  if (child)
    return child->CollectFirstAccessTasks(buffer, is_write, wg_id, phase,
                                          result);
  return false;
}

bool ScheduleUnit::CollectLastAccessTasks(
    const Buffer &buffer, bool is_write, int wg_id, SchedulePhase phase,
    std::set<const TaskNode *> &result) const {
  if (child)
    return child->CollectLastAccessTasks(buffer, is_write, wg_id, phase,
                                         result);
  return false;
}

bool WrapperNode::CollectFirstAccessTasks(
    const Buffer &buffer, bool is_write, int wg_id, SchedulePhase phase,
    std::set<const TaskNode *> &result) const {
  if (task) {
    if (task->CollectFirstAccessTasks(buffer, is_write, wg_id, phase, result))
      return true;
  }
  if (child) {
    if (child->CollectFirstAccessTasks(buffer, is_write, wg_id, phase, result))
      return true;
  }
  return false;
}

bool WrapperNode::CollectLastAccessTasks(
    const Buffer &buffer, bool is_write, int wg_id, SchedulePhase phase,
    std::set<const TaskNode *> &result) const {
  if (child) {
    if (child->CollectLastAccessTasks(buffer, is_write, wg_id, phase, result))
      return true;
  }
  if (task) {
    if (task->CollectLastAccessTasks(buffer, is_write, wg_id, phase, result))
      return true;
  }
  return false;
}

bool IfNode::CollectFirstAccessTasks(const Buffer &buffer, bool is_write,
                                     int wg_id, SchedulePhase phase,
                                     std::set<const TaskNode *> &result) const {
  if (task) {
    if (task->CollectFirstAccessTasks(buffer, is_write, wg_id, phase, result))
      return true;
  }
  bool then_must = false, else_must = false;
  if (then_child)
    then_must = then_child->CollectFirstAccessTasks(buffer, is_write, wg_id,
                                                    phase, result);
  if (else_child)
    else_must = else_child->CollectFirstAccessTasks(buffer, is_write, wg_id,
                                                    phase, result);
  return then_must && else_must;
}

bool IfNode::CollectLastAccessTasks(const Buffer &buffer, bool is_write,
                                    int wg_id, SchedulePhase phase,
                                    std::set<const TaskNode *> &result) const {
  bool then_must = false, else_must = false;
  if (then_child)
    then_must = then_child->CollectLastAccessTasks(buffer, is_write, wg_id,
                                                   phase, result);
  if (else_child)
    else_must = else_child->CollectLastAccessTasks(buffer, is_write, wg_id,
                                                   phase, result);
  if (then_must && else_must)
    return true;
  if (task) {
    if (task->CollectLastAccessTasks(buffer, is_write, wg_id, phase, result))
      return true;
  }
  return false;
}

static bool LoopMustExecute(const ControlNode *ctrl) {
  if (!ctrl->control.defined())
    return false;
  const ForNode *for_node = ctrl->control.get();
  const int64_t *extent_ptr = as_const_int(for_node->extent);
  if (extent_ptr && *extent_ptr >= 1) {
    if (for_node->step.has_value()) {
      const int64_t *step_ptr = as_const_int(for_node->step.value());
      if (step_ptr && *step_ptr >= 1)
        return true;
      if (!step_ptr)
        return false;
    } else {
      return true;
    }
  }
  return false;
}

bool ControlNode::CollectFirstAccessTasks(
    const Buffer &buffer, bool is_write, int wg_id, SchedulePhase phase,
    std::set<const TaskNode *> &result) const {
  if (task) {
    if (task->CollectFirstAccessTasks(buffer, is_write, wg_id, phase, result))
      return true;
  }
  bool body_must = false;
  if (child && child->IsSequence()) {
    auto *seq = static_cast<const SequenceNode *>(child.get());
    std::vector<const IRStructure *> ordered;
    ordered.reserve(seq->children.size());
    for (const auto &c : seq->children)
      ordered.push_back(c.get());
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const IRStructure *a, const IRStructure *b) {
                       auto *ua = dynamic_cast<const ScheduleUnit *>(a);
                       auto *ub = dynamic_cast<const ScheduleUnit *>(b);
                       if (ua && ub)
                         return ua->stage > ub->stage;
                       return false;
                     });
    body_must = SequenceCollectFirstAccessTasks(ordered, buffer, is_write,
                                                wg_id, phase, result);
  } else if (child) {
    body_must =
        child->CollectFirstAccessTasks(buffer, is_write, wg_id, phase, result);
  }
  if (body_must && LoopMustExecute(this))
    return true;
  return false;
}

bool ControlNode::CollectLastAccessTasks(
    const Buffer &buffer, bool is_write, int wg_id, SchedulePhase phase,
    std::set<const TaskNode *> &result) const {
  bool body_must = false;
  if (child && child->IsSequence()) {
    auto *seq = static_cast<const SequenceNode *>(child.get());
    std::vector<const IRStructure *> ordered;
    ordered.reserve(seq->children.size());
    for (const auto &c : seq->children)
      ordered.push_back(c.get());
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const IRStructure *a, const IRStructure *b) {
                       auto *ua = dynamic_cast<const ScheduleUnit *>(a);
                       auto *ub = dynamic_cast<const ScheduleUnit *>(b);
                       if (ua && ub)
                         return ua->stage > ub->stage;
                       return false;
                     });
    body_must = SequenceCollectLastAccessTasks(ordered, buffer, is_write, wg_id,
                                               phase, result);
  } else if (child) {
    body_must =
        child->CollectLastAccessTasks(buffer, is_write, wg_id, phase, result);
  }
  if (body_must && LoopMustExecute(this))
    return true;
  if (task) {
    if (task->CollectLastAccessTasks(buffer, is_write, wg_id, phase, result))
      return true;
  }
  return false;
}

bool SequenceNode::CollectFirstAccessTasks(
    const Buffer &buffer, bool is_write, int wg_id, SchedulePhase phase,
    std::set<const TaskNode *> &result) const {
  std::vector<const IRStructure *> ordered;
  ordered.reserve(children.size());
  for (const auto &c : children)
    ordered.push_back(c.get());
  return SequenceCollectFirstAccessTasks(ordered, buffer, is_write, wg_id,
                                         phase, result);
}

bool SequenceNode::CollectLastAccessTasks(
    const Buffer &buffer, bool is_write, int wg_id, SchedulePhase phase,
    std::set<const TaskNode *> &result) const {
  std::vector<const IRStructure *> ordered;
  ordered.reserve(children.size());
  for (const auto &c : children)
    ordered.push_back(c.get());
  return SequenceCollectLastAccessTasks(ordered, buffer, is_write, wg_id, phase,
                                        result);
}

} // namespace tl
} // namespace tvm
