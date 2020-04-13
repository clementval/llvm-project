//===-- lib/Semantics/check-acc-structure.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-acc-structure.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include <unordered_map>



#define CHECK_SIMPLE_CLAUSE(X, Y) \
void AccStructureChecker::Enter(const parser::AccClause::X &) { \
  CheckAllowed(AccClause::Y); \
}

#define CHECK_REQ_SCALAR_INT_CONSTANT_CLAUSE(X, Y) \
void AccStructureChecker::Enter(const parser::AccClause::X &c) { \
  CheckAllowed(AccClause::Y); \
  RequiresConstantPositiveParameter(AccClause::Y, c.v); \
}

namespace Fortran::semantics {

static constexpr inline AccClauseSet
    parallelAndKernelsOnlyAllowedAfterDeviceTypeClauses{
    AccClause::ASYNC, AccClause::WAIT, AccClause::NUM_GANGS,
    AccClause::NUM_WORKERS, AccClause::VECTOR_LENGTH};

static constexpr inline AccClauseSet serialOnlyAllowedAfterDeviceTypeClauses{
    AccClause::ASYNC, AccClause::WAIT};

static constexpr inline AccClauseSet loopOnlyAllowedAfterDeviceTypeClauses{
    AccClause::AUTO, AccClause::COLLAPSE, AccClause::INDEPENDENT,
    AccClause::GANG, AccClause::SEQ, AccClause::TILE, AccClause::VECTOR,
    AccClause::WORKER};

static constexpr inline AccClauseSet updateOnlyAllowedAfterDeviceTypeClauses{
    AccClause::ASYNC, AccClause::WAIT};

static constexpr inline AccClauseSet routineOnlyAllowedAfterDeviceTypeClauses{
    AccClause::BIND, AccClause::GANG, AccClause::VECTOR, AccClause::WORKER};

class NoBranchingEnforce {
public:
  NoBranchingEnforce(SemanticsContext &context,
      parser::CharBlock sourcePosition, AccDirective directive)
      : context_{context}, sourcePosition_{sourcePosition},
      currentDirective_{directive} {}
  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> void Post(const T &) {}

  template<typename T> bool Pre(const parser::Statement<T> &statement) {
    currentStatementSourcePosition_ = statement.source;
    return true;
  }

  void Post(const parser::ReturnStmt &) { emitBranchOutError("RETURN"); }
  void Post(const parser::ExitStmt &) { emitBranchOutError("EXIT"); }
  void Post(const parser::StopStmt &) { emitBranchOutError("STOP"); }

private:
  parser::MessageFixedText GetEnclosingMsg() {
    return "Enclosing block construct"_en_US;
  }

  void emitBranchOutError(const char* stmt) {
    context_.Say(currentStatementSourcePosition_,
        "%s statement is not allowed in a %s construct"_err_en_US, stmt,
        EnumToString(currentDirective_))
        .Attach(sourcePosition_, GetEnclosingMsg());
  }

  SemanticsContext &context_;
  parser::CharBlock currentStatementSourcePosition_;
  parser::CharBlock sourcePosition_;
  AccDirective currentDirective_;
};

void AccStructureChecker::PushContextAndClause(const parser::CharBlock &source,
                                               AccDirective dir)
{
  accContext_.emplace_back(source, dir);
  accContext_.back().allowedClauses = directiveClausesTable[dir].allowed;
  accContext_.back().allowedOnceClauses =
      directiveClausesTable[dir].allowedOnce;
  accContext_.back().allowedExclusiveClauses =
      directiveClausesTable[dir].allowedExclusive;
  accContext_.back().requiredOneOfClauses =
      directiveClausesTable[dir].requiredOneOf;
}

void AccStructureChecker::Enter(const parser::OpenACCConstruct &) {
  return;
}

void AccStructureChecker::Enter(const parser::OpenACCDeclarativeConstruct &) {
  return;
}

void AccStructureChecker::Enter(const parser::AccClause &x) {
  SetContextClause(x);
}

void AccStructureChecker::Leave(const parser::AccClauseList &) {
}

void AccStructureChecker::Enter(const parser::OpenACCBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::AccBeginBlockDirective>(x.t)};
  const auto &endBlockDir{std::get<parser::AccEndBlockDirective>(x.t)};
  const auto &beginDir{
      CheckMatching<parser::AccBlockDirective>(beginBlockDir, endBlockDir)};
  switch (beginDir.v) {
    case parser::AccBlockDirective::Directive::Parallel: {
      PushContextAndClause(beginDir.source, AccDirective::PARALLEL);
    } break;
    case parser::AccBlockDirective::Directive::Data: {
      PushContextAndClause(beginDir.source, AccDirective::DATA);
    } break;
    case parser::AccBlockDirective::Directive::Kernels: {
      PushContextAndClause(beginDir.source, AccDirective::KERNELS);
    } break;
    case parser::AccBlockDirective::Directive::Serial: {
      PushContextAndClause(beginDir.source, AccDirective::SERIAL);
    } break;
    case parser::AccBlockDirective::Directive::HostData: {
      PushContextAndClause(beginDir.source, AccDirective::HOST_DATA);
    } break;
  }
}

void AccStructureChecker::Leave(const parser::OpenACCBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::AccBeginBlockDirective>(x.t)};
  const auto &beginDir{std::get<parser::AccBlockDirective>(beginBlockDir.t)};
  const parser::Block &block{std::get<parser::Block>(x.t)};
  switch (beginDir.v) {
    case parser::AccBlockDirective::Directive::Kernels:
    case parser::AccBlockDirective::Directive::Parallel: {
      // Restriction - 880-881 (KERNELS)
      // Restriction - 843-844 (PARALLEL)
      CheckOnlyAllowedAfter(AccClause::DEVICE_TYPE,
          parallelAndKernelsOnlyAllowedAfterDeviceTypeClauses);
      // Restriction - 877 (KERNELS)
      // Restriction - 840 (PARALLEL)
      CheckNoBranching(block, GetContext().directive, beginDir.source);
    } break;
    case parser::AccBlockDirective::Directive::Serial: {
      // Restriction - 919
      CheckOnlyAllowedAfter(AccClause::DEVICE_TYPE,
          serialOnlyAllowedAfterDeviceTypeClauses);
      // Restriction - 916
      CheckNoBranching(block, AccDirective::SERIAL, beginDir.source);
    } break;
    case parser::AccBlockDirective::Directive::Data: {
      // Restriction - 1117-1118
      CheckRequireAtLeastOneOf();
    } break;
    case parser::AccBlockDirective::Directive::HostData: {
      // Restriction - 1578
      CheckRequireAtLeastOneOf();
    } break;
  }
  accContext_.pop_back();
}

void AccStructureChecker::CheckNoBranching(const parser::Block &block,
                      const AccDirective directive,
                      const parser::CharBlock &directiveSource) const {
  NoBranchingEnforce noBranchingEnforce{context_, directiveSource, directive};
  parser::Walk(block, noBranchingEnforce);
}

void AccStructureChecker::Enter(
    const parser::OpenACCStandaloneDeclarativeConstruct &x)
{
  const auto &dir{std::get<parser::AccDeclarativeDirective>(x.t)};
  switch (dir.v) {
    case parser::AccDeclarativeDirective::Directive::Declare: {
      PushContextAndClause(dir.source, AccDirective::DECLARE);
    } break;
  }
}

void AccStructureChecker::Leave(
    const parser::OpenACCStandaloneDeclarativeConstruct &)
{
  // Restriction - 2075
  CheckAtLeastOneClause();
  accContext_.pop_back();
}

void AccStructureChecker::Enter(const parser::OpenACCCombinedConstruct &x) {
  const auto &beginBlockDir{std::get<parser::AccBeginCombinedDirective>(x.t)};
  const auto &beginDir{std::get<parser::AccCombinedDirective>(beginBlockDir.t)};
  switch (beginDir.v) {
    case parser::AccCombinedDirective::Directive::KernelsLoop: {
      PushContextAndClause(beginDir.source, AccDirective::KERNELS_LOOP);
    } break;
    case parser::AccCombinedDirective::Directive::ParallelLoop: {
      PushContextAndClause(beginDir.source, AccDirective::PARALLEL_LOOP);
    } break;
    case parser::AccCombinedDirective::Directive::SerialLoop: {
      PushContextAndClause(beginDir.source, AccDirective::SERIAL_LOOP);
    } break;
  }
}

void AccStructureChecker::Leave(const parser::OpenACCCombinedConstruct &x) {
  const auto &beginBlockDir{std::get<parser::AccBeginCombinedDirective>(x.t)};
  const auto &beginDir{std::get<parser::AccCombinedDirective>(beginBlockDir.t)};
  switch (beginDir.v) {
    case parser::AccCombinedDirective::Directive::KernelsLoop:
    case parser::AccCombinedDirective::Directive::ParallelLoop: {
      // Restriction - 1962 -> (880-881) (KERNELS LOOP)
      // Restriction - 1962 -> (843-844) (PARALLEL LOOP)
      CheckOnlyAllowedAfter(AccClause::DEVICE_TYPE, {AccClause::ASYNC,
                                                     AccClause::WAIT,
                                                     AccClause::NUM_GANGS,
                                                     AccClause::NUM_WORKERS,
                                                     AccClause::VECTOR_LENGTH});
    } break;
    case parser::AccCombinedDirective::Directive::SerialLoop: {
      // Restriction - 1962 -> (919) (SERIAL LOOP)
      CheckOnlyAllowedAfter(AccClause::DEVICE_TYPE, {AccClause::ASYNC,
                                                     AccClause::WAIT});
    } break;
  }
  accContext_.pop_back();
}

std::string AccStructureChecker::ContextDirectiveAsFortran() {
  if(GetContext().directive == AccDirective::HOST_DATA)
    return EnumToString(GetContext().directive);
  auto dir{EnumToString(GetContext().directive)};
  std::replace(dir.begin(), dir.end(), '_', ' ');
  return dir;
}

void AccStructureChecker::Enter(const parser::OpenACCLoopConstruct &x) {
  const auto &beginDir{std::get<parser::AccBeginLoopDirective>(x.t)};
  const auto &dir{std::get<parser::AccLoopDirective>(beginDir.t)};
  switch (dir.v) {
    case parser::AccLoopDirective::Directive::Loop: {
      PushContextAndClause(dir.source, AccDirective::LOOP);
    } break;
  }
}

void AccStructureChecker::Leave(const parser::OpenACCLoopConstruct &x) {
  const auto &beginDir{std::get<parser::AccBeginLoopDirective>(x.t)};
  const auto &dir{std::get<parser::AccLoopDirective>(beginDir.t)};
  switch (dir.v) {
    case parser::AccLoopDirective::Directive::Loop: {
      // Restriction - 1615-1616
      CheckOnlyAllowedAfter(AccClause::DEVICE_TYPE,
          loopOnlyAllowedAfterDeviceTypeClauses);
      // Restriction - 1622
      CheckNotAllowedIfClause(AccClause::SEQ, {AccClause::GANG,
          AccClause::VECTOR, AccClause::WORKER});
    } break;
    default: {}
      break;
  }
  accContext_.pop_back();
}


void AccStructureChecker::Enter(const parser::OpenACCStandaloneConstruct &x) {
  const auto &dir{std::get<parser::AccStandaloneDirective>(x.t)};
  switch (dir.v) {
    case parser::AccStandaloneDirective::Directive::Init: {
      PushContextAndClause(dir.source, AccDirective::INIT);
    } break;
    case parser::AccStandaloneDirective::Directive::Cache: {
      PushContextAndClause(dir.source, AccDirective::CACHE);
    } break;
    case parser::AccStandaloneDirective::Directive::EnterData: {
      PushContextAndClause(dir.source, AccDirective::ENTER_DATA);
    } break;
    case parser::AccStandaloneDirective::Directive::ExitData: {
      PushContextAndClause(dir.source, AccDirective::EXIT_DATA);
    } break;
    case parser::AccStandaloneDirective::Directive::Set: {
      PushContextAndClause(dir.source, AccDirective::SET);
    } break;
    case parser::AccStandaloneDirective::Directive::Shutdown: {
      PushContextAndClause(dir.source, AccDirective::SHUTDOWN);
    } break;
    case parser::AccStandaloneDirective::Directive::Update: {
      PushContextAndClause(dir.source, AccDirective::UPDATE);
    } break;
  }
}

void AccStructureChecker::Leave(const parser::OpenACCStandaloneConstruct &x) {
  const auto &dir{std::get<parser::AccStandaloneDirective>(x.t)};
  switch (dir.v) {
    case parser::AccStandaloneDirective::Directive::EnterData:
    case parser::AccStandaloneDirective::Directive::ExitData:
    case parser::AccStandaloneDirective::Directive::Set: {
      // Restriction - 1117-1118 (ENTER DATA)
      // Restriction - 1161-1162 (EXIT DATA)
      // Restriction - 2254 (SET)
      CheckRequireAtLeastOneOf();
    } break;
    case parser::AccStandaloneDirective::Directive::Update: {
      // Restriction - 2301
      CheckOnlyAllowedAfter(AccClause::DEVICE_TYPE,
          updateOnlyAllowedAfterDeviceTypeClauses);
    } break;
    default: {}
      break;
  }
  accContext_.pop_back();
}

void AccStructureChecker::Enter(const parser::OpenACCRoutineConstruct &x) {
  PushContextAndClause(x.source, AccDirective::ROUTINE);
}
void AccStructureChecker::Leave(const parser::OpenACCRoutineConstruct &) {
  // Restriction - 2409
  CheckRequireAtLeastOneOf();
  // Restriction - 2407-2408
  CheckOnlyAllowedAfter(AccClause::DEVICE_TYPE,
      routineOnlyAllowedAfterDeviceTypeClauses);
  accContext_.pop_back();
}

// Clause checkers
CHECK_REQ_SCALAR_INT_CONSTANT_CLAUSE(Collapse, COLLAPSE)

CHECK_SIMPLE_CLAUSE(Auto, AUTO)
CHECK_SIMPLE_CLAUSE(Async, ASYNC)
CHECK_SIMPLE_CLAUSE(Attach, ATTACH)
CHECK_SIMPLE_CLAUSE(Capture, CAPTURE)
CHECK_SIMPLE_CLAUSE(Copy, COPY)
CHECK_SIMPLE_CLAUSE(Bind, BIND)
CHECK_SIMPLE_CLAUSE(Default, DEFAULT)
CHECK_SIMPLE_CLAUSE(DefaultAsync, DEFAULT_ASYNC)
CHECK_SIMPLE_CLAUSE(Delete, DELETE)
CHECK_SIMPLE_CLAUSE(Detach, DETACH)
CHECK_SIMPLE_CLAUSE(Device, DEVICE)
CHECK_SIMPLE_CLAUSE(DeviceNum, DEVICE_NUM)
CHECK_SIMPLE_CLAUSE(DevicePtr, DEVICEPTR)
CHECK_SIMPLE_CLAUSE(DeviceResident, DEVICE_RESIDENT)
CHECK_SIMPLE_CLAUSE(DeviceType, DEVICE_TYPE)
CHECK_SIMPLE_CLAUSE(Finalize, FINALIZE)
CHECK_SIMPLE_CLAUSE(FirstPrivate, FIRSTPRIVATE)
CHECK_SIMPLE_CLAUSE(Gang, GANG)
CHECK_SIMPLE_CLAUSE(Host, HOST)
CHECK_SIMPLE_CLAUSE(If, IF)
CHECK_SIMPLE_CLAUSE(IfPresent, IF_PRESENT)
CHECK_SIMPLE_CLAUSE(Independent, INDEPENDENT)
CHECK_SIMPLE_CLAUSE(Link, LINK)
CHECK_SIMPLE_CLAUSE(NoCreate, NO_CREATE)
CHECK_SIMPLE_CLAUSE(NoHost, NOHOST)
CHECK_SIMPLE_CLAUSE(NumGangs, NUM_GANGS)
CHECK_SIMPLE_CLAUSE(NumWorkers, NUM_WORKERS)
CHECK_SIMPLE_CLAUSE(Present, PRESENT)
CHECK_SIMPLE_CLAUSE(Private, PRIVATE)
CHECK_SIMPLE_CLAUSE(Read, READ)
CHECK_SIMPLE_CLAUSE(Reduction, REDUCTION)
CHECK_SIMPLE_CLAUSE(Self, SELF)
CHECK_SIMPLE_CLAUSE(Seq, SEQ)
CHECK_SIMPLE_CLAUSE(Tile, TILE)
CHECK_SIMPLE_CLAUSE(UseDevice, USE_DEVICE)
CHECK_SIMPLE_CLAUSE(Vector, VECTOR)
CHECK_SIMPLE_CLAUSE(VectorLength, VECTOR_LENGTH)
CHECK_SIMPLE_CLAUSE(Wait, WAIT)
CHECK_SIMPLE_CLAUSE(Worker, WORKER)
CHECK_SIMPLE_CLAUSE(Write, WRITE)

void AccStructureChecker::Enter(const parser::AccClause::Create &c) {
  CheckAllowed(AccClause::CREATE);
  const auto &modifierClause{c.v};
  if (const auto &modifier{std::get<std::optional<parser::AccDataModifier>>(
      modifierClause.t)})
  {
    if(modifier->v != parser::AccDataModifier::Modifier::Zero) {
      context_.Say(GetContext().clauseSource,
          "Only the ZERO modifier is allowed for the %s clause "
          "on the %s directive"_err_en_US, EnumToString(AccClause::CREATE),
          ContextDirectiveAsFortran());
    }
  }
}

void AccStructureChecker::Enter(const parser::AccClause::Copyin &c) {
  CheckAllowed(AccClause::COPYIN);
  const auto &modifierClause{c.v};
  if (const auto &modifier{std::get<std::optional<parser::AccDataModifier>>(
      modifierClause.t)})
  {
    if(modifier->v != parser::AccDataModifier::Modifier::ReadOnly) {
      context_.Say(GetContext().clauseSource,
          "Only the READONLY modifier is allowed for the %s clause "
          "on the %s directive"_err_en_US, EnumToString(AccClause::COPYIN),
          ContextDirectiveAsFortran());
    }
  }
}

void AccStructureChecker::Enter(const parser::AccClause::Copyout &c) {
  CheckAllowed(AccClause::COPYOUT);
  const auto &modifierClause{c.v};
  if (const auto &modifier{std::get<std::optional<parser::AccDataModifier>>(
      modifierClause.t)})
  {
    if(modifier->v != parser::AccDataModifier::Modifier::Zero) {
      context_.Say(GetContext().clauseSource,
          "Only the ZERO modifier is allowed for the %s clause "
          "on the %s directive"_err_en_US, EnumToString(AccClause::COPYOUT),
          ContextDirectiveAsFortran());
    }
  }
}

void AccStructureChecker::CheckAllowed(AccClause type) {
  if (!GetContext().allowedClauses.test(type) &&
      !GetContext().allowedOnceClauses.test(type) &&
      !GetContext().allowedExclusiveClauses.test(type) &&
      !GetContext().requiredOneOfClauses.test(type))
  {
    context_.Say(GetContext().clauseSource,
        "%s clause is not allowed on the %s directive"_err_en_US,
        EnumToString(type), parser::ToUpperCaseLetters(
        GetContext().directiveSource.ToString()));
    return;
  }
  if ((GetContext().allowedOnceClauses.test(type) ||
       GetContext().allowedExclusiveClauses.test(type)) &&
      FindClause(type))
  {
    context_.Say(GetContext().clauseSource,
        "At most one %s clause can appear on the %s directive"_err_en_US,
        EnumToString(type), parser::ToUpperCaseLetters(
            GetContext().directiveSource.ToString()));
    return;
  }
  if (GetContext().allowedExclusiveClauses.test(type)) {
    std::vector<AccClause> others;
    GetContext().allowedExclusiveClauses.IterateOverMembers([&](AccClause o) {
      if (FindClause(o)) {
        others.emplace_back(o);
      }
    });
    for (const auto &e : others) {
      context_.Say(GetContext().clauseSource,
          "%s and %s clauses are mutually exclusive and may not appear on the "
          "same %s directive"_err_en_US, EnumToString(type), EnumToString(e),
          parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
    }
    if (!others.empty()) {
      return;
    }
  }
  SetContextClauseInfo(type);
  AddClauseToCrtContext(type);
}

void AccStructureChecker::CheckOnlyAllowedAfter(AccClause clause,
    AccClauseSet set) {
  bool enforceCheck = false;
  for (auto cl : GetContext().actualClauses) {
    if(cl == clause) {
      enforceCheck = true;
      continue;
    } else if(enforceCheck && !set.test(cl)) {
      auto parserClause = GetContext().clauseInfo.find(cl);
      context_.Say(parserClause->second->source,
          "Clause %s is not allowed after clause %s on the %s "
          "directive"_err_en_US,
          EnumToString(cl), EnumToString(clause),
          ContextDirectiveAsFortran());
    }
  }
}

void AccStructureChecker::CheckRequireAtLeastOneOf() {
  for (auto cl : GetContext().actualClauses) {
    if(GetContext().requiredOneOfClauses.test(cl))
      return;
  }
  // No clause matched in the actual clauses list
  context_.Say(GetContext().directiveSource,
      "At least one of %s clause must appear on the %s directive"_err_en_US,
      ClauseSetToString(GetContext().requiredOneOfClauses),
      ContextDirectiveAsFortran());
}

void AccStructureChecker::CheckAtLeastOneClause() {
  if(GetContext().actualClauses.empty()) {
    context_.Say(GetContext().directiveSource,
        "At least one clause is required on the %s directive"_err_en_US,
        ContextDirectiveAsFortran());
  }
}

/**
 * Enforce restriction where clauses in the given set are not allowed if the
 * given clause appears.
 */
void AccStructureChecker::CheckNotAllowedIfClause(AccClause clause,
    AccClauseSet set) {
  if(std::find(GetContext().actualClauses.begin(),
      GetContext().actualClauses.end(), clause) ==
      GetContext().actualClauses.end()) {
    return; // Clause is not present
  }

  for (auto cl : GetContext().actualClauses) {
    if(set.test(cl)) {
      context_.Say(GetContext().directiveSource,
          "Clause %s is not allowed if clause %s appears on the %s directive"_err_en_US,
          EnumToString(cl), EnumToString(clause),
          ContextDirectiveAsFortran());
    }
  }
}


void AccStructureChecker::RequiresConstantPositiveParameter(
    const AccClause &clause, const parser::ScalarIntConstantExpr &i)
{
  if (const auto v{GetIntValue(i)}) {
    if (*v <= 0) {
      context_.Say(GetContext().clauseSource,
          "The parameter of the %s clause on the %s directive must be "
          "a constant positive integer expression"_err_en_US,
          EnumToString(clause), ContextDirectiveAsFortran());
    }
  }
}

void AccStructureChecker::OptionalConstantPositiveParameter(
    const AccClause &clause,
    const std::optional<parser::ScalarIntConstantExpr> &o)
{
  if(o != std::nullopt) {
    RequiresConstantPositiveParameter(clause, o.value());
  }
}

std::string AccStructureChecker::ClauseSetToString(const AccClauseSet set) {
  std::string list;
  set.IterateOverMembers([&](AccClause o) {
      if(!list.empty())
        list.append(", ");
      list.append(EnumToString(o));
  });
  return list;
}

void AccStructureChecker::SayNotMatching(
    const parser::CharBlock &beginSource, const parser::CharBlock &endSource)
{
  context_
      .Say(endSource, "Unmatched %s directive"_err_en_US,
           parser::ToUpperCaseLetters(endSource.ToString()))
      .Attach(beginSource, "Does not match directive"_en_US);
}

} // namespace Fortran::semantics
