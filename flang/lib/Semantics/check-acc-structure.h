//===-- lib/Semantics/check-acc-structure.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// OpenACC structure validity check list
//    1. invalid clauses on directive
//    2. invalid repeated clauses on directive
//    3. invalid nesting of regions
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_ACC_STRUCTURE_H_
#define FORTRAN_SEMANTICS_CHECK_ACC_STRUCTURE_H_

#include "flang/Common/enum-set.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include <unordered_map>

namespace Fortran::semantics {

ENUM_CLASS(AccDirective, ATOMIC, CACHE, DATA, DECLARE, ENTER_DATA, EXIT_DATA,
           INIT, HOST_DATA, KERNELS, KERNELS_LOOP, LOOP, PARALLEL,
           PARALLEL_LOOP, ROUTINE, SERIAL, SERIAL_LOOP, SET, SHUTDOWN, UPDATE,
           WAIT)

using AccDirectiveSet = common::EnumSet<AccDirective, AccDirective_enumSize>;

ENUM_CLASS(AccClause, AUTO, ASYNC, ATTACH, CAPTURE, BIND, COLLAPSE, COPY,
           COPYIN, COPYOUT, DEFAULT, DEFAULT_ASYNC, DELETE, CREATE, DETACH,
           DEVICE, DEVICE_NUM, DEVICEPTR, DEVICE_RESIDENT, DEVICE_TYPE,
           FINALIZE, FIRSTPRIVATE, GANG, HOST, IF, IF_PRESENT, INDEPENDENT,
           LINK, NO_CREATE, NOHOST, NUM_GANGS, NUM_WORKERS, PRESENT, PRIVATE,
           READ, REDUCTION, TILE, USE_DEVICE, VECTOR_LENGTH, SELF, SEQ, VECTOR,
           WAIT, WORKER, WRITE)

using AccClauseSet = common::EnumSet<AccClause, AccClause_enumSize>;

class AccStructureChecker : public virtual BaseChecker {
public:
  AccStructureChecker(SemanticsContext &context) : context_{context} {}

  // Construct and directives
  void Enter(const parser::OpenACCBlockConstruct &);
  void Leave(const parser::OpenACCBlockConstruct &);
  void Enter(const parser::OpenACCConstruct &);
  void Enter(const parser::OpenACCCombinedConstruct &);
  void Leave(const parser::OpenACCCombinedConstruct &);
  void Enter(const parser::OpenACCDeclarativeConstruct &);
  void Enter(const parser::OpenACCLoopConstruct &);
  void Leave(const parser::OpenACCLoopConstruct &);
  void Enter(const parser::OpenACCRoutineConstruct &);
  void Leave(const parser::OpenACCRoutineConstruct &);
  void Enter(const parser::OpenACCStandaloneConstruct &);
  void Leave(const parser::OpenACCStandaloneConstruct &);
  void Enter(const parser::OpenACCStandaloneDeclarativeConstruct &);
  void Leave(const parser::OpenACCStandaloneDeclarativeConstruct &);

  // Clauses
  void Leave(const parser::AccClauseList &);
  void Enter(const parser::AccClause &);
  void Enter(const parser::AccClause::Auto &);
  void Enter(const parser::AccClause::Async &);
  void Enter(const parser::AccClause::Attach &);
  void Enter(const parser::AccClause::Capture &);
  void Enter(const parser::AccClause::Bind &);
  void Enter(const parser::AccClause::Collapse &);
  void Enter(const parser::AccClause::Copy &);
  void Enter(const parser::AccClause::Copyin &);
  void Enter(const parser::AccClause::Copyout &);
  void Enter(const parser::AccClause::Create &);
  void Enter(const parser::AccClause::Default &);
  void Enter(const parser::AccClause::DefaultAsync &);
  void Enter(const parser::AccClause::Delete &);
  void Enter(const parser::AccClause::Detach &);
  void Enter(const parser::AccClause::Device &);
  void Enter(const parser::AccClause::DeviceNum &);
  void Enter(const parser::AccClause::DevicePtr &);
  void Enter(const parser::AccClause::DeviceType &);
  void Enter(const parser::AccClause::DeviceResident &);
  void Enter(const parser::AccClause::Finalize &);
  void Enter(const parser::AccClause::FirstPrivate &);
  void Enter(const parser::AccClause::Gang &);
  void Enter(const parser::AccClause::Host &);
  void Enter(const parser::AccClause::If &);
  void Enter(const parser::AccClause::IfPresent &);
  void Enter(const parser::AccClause::Independent&);
  void Enter(const parser::AccClause::Link&);
  void Enter(const parser::AccClause::NoCreate &);
  void Enter(const parser::AccClause::NoHost &);
  void Enter(const parser::AccClause::NumGangs &);
  void Enter(const parser::AccClause::NumWorkers &);
  void Enter(const parser::AccClause::Present &);
  void Enter(const parser::AccClause::Private &);
  void Enter(const parser::AccClause::Read &);
  void Enter(const parser::AccClause::Reduction &);
  void Enter(const parser::AccClause::Self &);
  void Enter(const parser::AccClause::Seq &);
  void Enter(const parser::AccClause::Tile &);
  void Enter(const parser::AccClause::UseDevice &);
  void Enter(const parser::AccClause::Vector &);
  void Enter(const parser::AccClause::VectorLength &);
  void Enter(const parser::AccClause::Worker &);
  void Enter(const parser::AccClause::Wait &);
  void Enter(const parser::AccClause::Write &);

private:
  struct AccDirectiveClauses {
    const AccClauseSet allowed;
    const AccClauseSet allowedOnce;
    const AccClauseSet allowedExclusive;
    const AccClauseSet requiredOneOf;
  };

  std::unordered_map<AccDirective, AccDirectiveClauses> directiveClausesTable =
      {
      {AccDirective::ATOMIC, { // 2.12
        {}, {}, {}, {}
      }},
      {AccDirective::CACHE, { // 2.10
        {},{},{},{}}},
      {AccDirective::DATA, { // 2.6.5
        {},
        {AccClause::IF},
        {},
        {AccClause::ATTACH, AccClause::COPY, AccClause::COPYIN,
         AccClause::COPYOUT, AccClause::CREATE, AccClause::DEFAULT,
         AccClause::DEVICEPTR, AccClause::NO_CREATE, AccClause::PRESENT}}},
      {AccDirective::DECLARE, { // 2.13
        {AccClause::COPY, AccClause::COPYIN, AccClause::COPYOUT,
         AccClause::CREATE, AccClause::PRESENT, AccClause::DEVICEPTR,
         AccClause::DEVICE_RESIDENT, AccClause::LINK},
        {},
        {},
        {}}},
      {AccDirective::ENTER_DATA, { // 2.14.6
        {},
        {AccClause::ASYNC, AccClause::IF, AccClause::WAIT},
        {},
        {AccClause::ATTACH, AccClause::CREATE, AccClause::COPYIN}}},
      {AccDirective::EXIT_DATA, { // 2.14.7
        {},
        {AccClause::ASYNC, AccClause::IF, AccClause::WAIT,
         AccClause::FINALIZE},
        {},
        {AccClause::COPYOUT, AccClause::DELETE, AccClause::DETACH}}},
      {AccDirective::HOST_DATA, { // 2.8
        {},
        {AccClause::IF, AccClause::IF_PRESENT},
        {},
        {AccClause::USE_DEVICE}}},
      {AccDirective::INIT, { // 2.14.1
        {},
        {AccClause::DEVICE_NUM, AccClause::DEVICE_TYPE, AccClause::IF},
        {},
        {}}},
      {AccDirective::KERNELS, { // 2.5.2
        {AccClause::ATTACH, AccClause::COPY, AccClause::COPYIN,
         AccClause::COPYOUT, AccClause::CREATE, AccClause::DEVICE_TYPE,
         AccClause::NO_CREATE, AccClause::PRESENT, AccClause::DEVICEPTR},
        {AccClause::ASYNC, AccClause::DEFAULT, AccClause::IF,
         AccClause::NUM_GANGS, AccClause::NUM_WORKERS, AccClause::SELF,
         AccClause::VECTOR_LENGTH, AccClause::WAIT},
        {},
        {}}},
      {AccDirective::KERNELS_LOOP, { // 2.11
        {AccClause::COPY, AccClause::COPYIN, AccClause::COPYOUT,
         AccClause::CREATE, AccClause::DEVICE_TYPE, AccClause::NO_CREATE,
         AccClause::PRESENT,  AccClause::PRIVATE, AccClause::DEVICEPTR,
         AccClause::ATTACH},
        {AccClause::ASYNC, AccClause::COLLAPSE, AccClause::DEFAULT,
         AccClause::GANG, AccClause::IF, AccClause::INDEPENDENT,
         AccClause::NUM_GANGS, AccClause::NUM_WORKERS, AccClause::REDUCTION,
         AccClause::SELF, AccClause::TILE, AccClause::VECTOR,
         AccClause::VECTOR_LENGTH, AccClause::WAIT, AccClause::WORKER},
        {AccClause::AUTO, AccClause::INDEPENDENT, AccClause::SEQ},
        {}}},
      {AccDirective::LOOP, { // 2.9
        {AccClause::DEVICE_TYPE, AccClause::PRIVATE},
        {AccClause::COLLAPSE, AccClause::GANG, AccClause::REDUCTION,
         AccClause::TILE, AccClause::VECTOR, AccClause::WORKER},
        {AccClause::AUTO, AccClause::INDEPENDENT, AccClause::SEQ},
        {}}},
      {AccDirective::PARALLEL, { // 2.5.1
        {AccClause::ATTACH, AccClause::COPY, AccClause::COPYIN,
         AccClause::COPYOUT, AccClause::CREATE, AccClause::DEVICEPTR,
         AccClause::DEVICE_TYPE, AccClause::NO_CREATE, AccClause::PRESENT,
         AccClause::PRIVATE, AccClause::FIRSTPRIVATE, AccClause::WAIT},
        {AccClause::ASYNC, AccClause::DEFAULT, AccClause::IF,
         AccClause::NUM_GANGS, AccClause::NUM_WORKERS, AccClause::REDUCTION,
         AccClause::SELF, AccClause::VECTOR_LENGTH},
        {},
        {}}},
      {AccDirective::PARALLEL_LOOP, { // 2.11
        {AccClause::ATTACH, AccClause::COPY, AccClause::COPYIN,
         AccClause::COPYOUT, AccClause::CREATE, AccClause::DEVICEPTR,
         AccClause::DEVICE_TYPE, AccClause::FIRSTPRIVATE,
         AccClause::NO_CREATE, AccClause::PRESENT, AccClause::PRIVATE,
         AccClause::TILE, AccClause::WAIT},
        {AccClause::ASYNC, AccClause::COLLAPSE, AccClause::DEFAULT,
         AccClause::GANG, AccClause::IF, AccClause::NUM_GANGS,
         AccClause::NUM_WORKERS, AccClause::REDUCTION, AccClause::SELF,
         AccClause::VECTOR, AccClause::VECTOR_LENGTH, AccClause::WORKER},
        {AccClause::AUTO, AccClause::INDEPENDENT, AccClause::SEQ},
        {}}
       },
      {AccDirective::ROUTINE, { // 2.15.1
        {},
        {AccClause::BIND, AccClause::DEVICE_TYPE, AccClause::NOHOST},
        {},
        {AccClause::GANG, AccClause::SEQ, AccClause::VECTOR,
         AccClause::WORKER}}},
      {AccDirective::SERIAL, { // 2.5.3
        {AccClause::ATTACH, AccClause::COPY, AccClause::COPYIN,
         AccClause::COPYOUT, AccClause::CREATE, AccClause::DEVICEPTR,
         AccClause::DEVICE_TYPE, AccClause::FIRSTPRIVATE,
         AccClause::NO_CREATE, AccClause::PRESENT, AccClause::PRIVATE,
         AccClause::WAIT},
        {AccClause::ASYNC, AccClause::DEFAULT, AccClause::IF,
         AccClause::REDUCTION, AccClause::SELF},
        {},
        {}}},
      {AccDirective::SERIAL_LOOP, { // 2.11
        {AccClause::ATTACH, AccClause::COPY, AccClause::COPYIN,
         AccClause::COPYOUT, AccClause::CREATE, AccClause::DEVICEPTR,
         AccClause::DEVICE_TYPE, AccClause::FIRSTPRIVATE,
         AccClause::NO_CREATE, AccClause::PRESENT, AccClause::PRIVATE,
         AccClause::WAIT},
        {AccClause::ASYNC, AccClause::COLLAPSE, AccClause::DEFAULT,
         AccClause::GANG, AccClause::IF, AccClause::REDUCTION, AccClause::SELF,
         AccClause::TILE, AccClause::VECTOR, AccClause::WORKER},
        {AccClause::AUTO, AccClause::INDEPENDENT, AccClause::SEQ},
        {}}},
      {AccDirective::SET, { // 2.14.3
        {},
        {AccClause::IF},
        {},
        {AccClause::DEFAULT_ASYNC, AccClause::DEVICE_NUM,
         AccClause::DEVICE_TYPE}}},
      {AccDirective::SHUTDOWN, { // 2.14.2
        {},
        {AccClause::DEVICE_NUM, AccClause::DEVICE_TYPE, AccClause::IF},
        {},
        {}}},
      {AccDirective::UPDATE, { // 2.14.4
        {AccClause::DEVICE_TYPE, AccClause::WAIT},
        {AccClause::ASYNC, AccClause::IF, AccClause::IF_PRESENT},
        {},
        {AccClause::DEVICE, AccClause::HOST, AccClause::SELF}}},
      {AccDirective::WAIT, { // 2.14.5
        {},
        {AccClause::ASYNC, AccClause::IF},
        {},
        {}}}
  };

  struct AccContext {
    AccContext(parser::CharBlock source, AccDirective d)
        : directiveSource{source}, directive{d} {}

    parser::CharBlock directiveSource{nullptr};
    parser::CharBlock clauseSource{nullptr};
    AccDirective directive;
    AccClauseSet allowedClauses{};
    AccClauseSet allowedOnceClauses{};
    AccClauseSet allowedExclusiveClauses{};
    AccClauseSet requiredOneOfClauses{};

    const parser::AccClause *clause{nullptr};
    std::multimap<AccClause, const parser::AccClause *> clauseInfo;
    std::list<AccClause> actualClauses;
  };

  // back() is the top of the stack
  AccContext &GetContext() {
    CHECK(!accContext_.empty());
    return accContext_.back();
  }

  void SetContextClause(const parser::AccClause &clause) {
    GetContext().clauseSource = clause.source;
    GetContext().clause = &clause;
  }

  void SetContextClauseInfo(AccClause type) {
    GetContext().clauseInfo.emplace(type, GetContext().clause);
  }

  void AddClauseToCrtContext(AccClause type) {
    GetContext().actualClauses.push_back(type);
  }

  const parser::AccClause *FindClause(AccClause type) {
    auto it{GetContext().clauseInfo.find(type)};
    if (it != GetContext().clauseInfo.end()) {
      return it->second;
    }
    return nullptr;
  }


  void PushContextAndClause(const parser::CharBlock &source, AccDirective dir);

  void SayNotMatching(const parser::CharBlock &, const parser::CharBlock &);

  template<typename A, typename B, typename C>
  const A &CheckMatching(const B &beginDir, const C &endDir) {
    const A &begin{std::get<A>(beginDir.t)};
    const A &end = endDir.v;
    if (begin.v != end.v) {
      SayNotMatching(begin.source, end.source);
    }
    return begin;
  }

  // Check that only clauses in set are after the specific clauses.
  void CheckOnlyAllowedAfter(AccClause clause, AccClauseSet set);
  void CheckRequireAtLeastOneOf();
  void CheckAllowed(AccClause);
  void CheckAtLeastOneClause();
  void CheckNotAllowedIfClause(AccClause clause, AccClauseSet set);
  std::string ContextDirectiveAsFortran();

  void CheckNoBranching(const parser::Block &block,
      const AccDirective directive,
      const parser::CharBlock &directiveSource) const;

  void RequiresConstantPositiveParameter(
      const AccClause &clause, const parser::ScalarIntConstantExpr &i);
  void OptionalConstantPositiveParameter(
      const AccClause &clause, const std::optional<parser::ScalarIntConstantExpr> &o);

  SemanticsContext &context_;
  std::vector<AccContext> accContext_;  // used as a stack

  std::string ClauseSetToString(const AccClauseSet set);
};

}

#endif // FORTRAN_SEMANTICS_CHECK_ACC_STRUCTURE_H_
