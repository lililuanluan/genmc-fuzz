/*
 * GenMC -- Generic Model Checking.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, you can access it online at
 * http://www.gnu.org/licenses/gpl-3.0.html.
 *
 * Author: Michalis Kokologiannakis <michalis@mpi-sws.org>
 */

#include "config.h"
#include "Config.hpp"
#include "DepExecutionGraph.hpp"
#include "DriverHandlerDispatcher.hpp"
#include "Error.hpp"
#include "LLVMModule.hpp"
#include "Logger.hpp"
#include "GenMCDriver.hpp"
#include "Interpreter.h"
#include "GraphIterators.hpp"
#include "LabelVisitor.hpp"
#include "MaximalIterator.hpp"
#include "Parser.hpp"
#include "SExprVisitor.hpp"
#include "ThreadPool.hpp"
#include <llvm/IR/Verifier.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_os_ostream.h>

#include <algorithm>
#include <csignal>
#include "Event.hpp"

#include "GraphCounter.hpp"

 /************************************************************
  ** GENERIC MODEL CHECKING DRIVER
  ***********************************************************/

GenMCDriver::GenMCDriver(std::shared_ptr<const Config> conf, std::unique_ptr<llvm::Module> mod,
	std::unique_ptr<ModuleInfo> modInfo, Mode mode /* = VerificationMode{} */)
	: userConf(std::move(conf)), mode(mode)
{
	/* Set up the execution context */
	auto execGraph = userConf->isDepTrackingModel ?
		std::make_unique<DepExecutionGraph>() :
		std::make_unique<ExecutionGraph>();
	execStack.emplace_back(std::move(execGraph), std::move(LocalQueueT()), std::move(ChoiceMap()));

	/* Create an interpreter for the program's instructions */
	std::string buf;
	EE = llvm::Interpreter::create(std::move(mod), std::move(modInfo), this, getConf(),
		getAddrAllocator(), &buf);

	/* Set up a random-number generator (for the scheduler) */
	std::random_device rd;
#ifdef DEBUG_LUAN
	// Print("setting up rand device");
	// getchar();
#endif
	auto seedVal = (!userConf->randomScheduleSeed.empty()) ?
		(MyRNG::result_type)stoull(userConf->randomScheduleSeed) : rd();
	if (userConf->printRandomScheduleSeed) {
		llvm::outs() << "Seed: " << seedVal << "\n";
	}
	rng.seed(seedVal);
#ifdef DEBUG_LUAN
	estRng.seed(seedVal);
#else
	estRng.seed(rd());
#endif


	/*
	 * Make sure we can resolve symbols in the program as well. We use 0
	 * as an argument in order to load the program, not a library. This
	 * is useful as it allows the executions of external functions in the
	 * user code.
	 */
	std::string ErrorStr;
	if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr, &ErrorStr)) {
		WARN("Could not resolve symbols in the program: " + ErrorStr);
	}
}

GenMCDriver::~GenMCDriver() = default;

GenMCDriver::Execution::Execution(std::unique_ptr<ExecutionGraph> g, LocalQueueT&& w, ChoiceMap&& m)
	: graph(std::move(g)), workqueue(std::move(w)), choices(std::move(m)) {}
GenMCDriver::Execution::~Execution() = default;


void repairRead(ExecutionGraph& g, ReadLabel* lab)
{
	auto last = (store_rbegin(g, lab->getAddr()) == store_rend(g, lab->getAddr())) ? Event::getInit() : store_rbegin(g, lab->getAddr())->getPos();
	g.changeRf(lab->getPos(), last);
	lab->setAddedMax(true);
	lab->setIPRStatus(g.getEventLabel(last)->getStamp() > lab->getStamp());
}

void repairDanglingReads(ExecutionGraph& g)
{
	for (auto i = 0U; i < g.getNumThreads(); i++) {
		auto* rLab = llvm::dyn_cast<ReadLabel>(g.getLastThreadLabel(i));
		if (!rLab)
			continue;
		if (!rLab->getRf()) {
			repairRead(g, rLab);
		}
	}
}

void GenMCDriver::Execution::restrictGraph(Stamp stamp)	// call this for rerun
{
	/* Restrict the graph (and relations). It can be the case that
	 * events with larger stamp remain in the graph (e.g.,
	 * BEGINs). Fix their stamps too. */
	auto& g = getGraph();
	g.cutToStamp(stamp);	// 
#ifdef FUZZ_LUAN
	// Print("after cut to stamp: \n", g);
#endif
	g.compressStampsAfter(stamp);
	repairDanglingReads(g);
}

void GenMCDriver::Execution::restrictWorklist(Stamp stamp)
{
	// Print("before restrict worklist:");
	// Print(GREEN("work queue: "));
	// for (auto&& p : workqueue) {
	// 	Print(p.first, ", ", p.second);
	// }

	std::vector<Stamp> idxsToRemove;

	auto& workqueue = getWorkqueue();
	for (auto rit = workqueue.rbegin(); rit != workqueue.rend(); ++rit)
		if (rit->first > stamp && rit->second.empty())
			idxsToRemove.push_back(rit->first); // TODO: break out of loop?

	for (auto& i : idxsToRemove)
		workqueue.erase(i);

	// Print("after restrict worklist:");
	// Print(GREEN("work queue: "));
	// for (auto&& p : workqueue) {
	// 	Print(p.first, ", ", p.second);
	// }
}

void GenMCDriver::Execution::restrictChoices(Stamp stamp)
{
	auto& choices = getChoiceMap();
	for (auto cit = choices.begin(); cit != choices.end(); ) {
		if (cit->first > stamp.get()) {
			cit = choices.erase(cit);
		}
		else {
			++cit;
		}
	}
}

void GenMCDriver::Execution::restrict(Stamp stamp)
{
#ifdef FUZZ_LUAN
	const auto& g = getGraph();
	Print("before restricting..., stamp = ", stamp);
	// Print(g);
#endif
	restrictGraph(stamp);
#ifdef FUZZ_LUAN
	// Print(GREEN("after restrictGraph..."));
	// Print(g);
	// getchar();
#endif
	restrictWorklist(stamp);
	restrictChoices(stamp);
#ifdef FUZZ_LUAN

	// getchar();
#endif
}

void GenMCDriver::pushExecution(Execution&& e)
{
	execStack.push_back(std::move(e));
}

bool GenMCDriver::popExecution()
{
	Print(GREEN("popExecution"));
	// getchar();
	if (execStack.empty())
		return false;
	execStack.pop_back();
	return !execStack.empty();
}

GenMCDriver::State::State(std::unique_ptr<ExecutionGraph> g, ChoiceMap&& m, SAddrAllocator&& a,
	llvm::BitVector&& fds, ValuePrefixT&& c, Event la)
	: graph(std::move(g)), choices(std::move(m)), alloctor(std::move(a)), fds(std::move(fds)), cache(std::move(c)), lastAdded(la) {}
GenMCDriver::State::~State() = default;

void GenMCDriver::initFromState(std::unique_ptr<State> s)
{
	execStack.clear();
	execStack.emplace_back(std::move(s->graph), LocalQueueT(), std::move(s->choices));
	alloctor = std::move(s->alloctor);
	fds = std::move(s->fds);
	seenPrefixes = std::move(s->cache);
	lastAdded = s->lastAdded;
}

std::unique_ptr<GenMCDriver::State>
GenMCDriver::extractState()
{
	auto cache = std::move(seenPrefixes);
	seenPrefixes.clear();
	return std::make_unique<State>(
		getGraph().clone(), ChoiceMap(getChoiceMap()), SAddrAllocator(alloctor),
		llvm::BitVector(fds), std::move(cache), lastAdded);
}

/* Returns a fresh address to be used from the interpreter */
SAddr GenMCDriver::getFreshAddr(const MallocLabel* aLab)
{
	/* The arguments to getFreshAddr() need to be well-formed;
	 * make sure the alignment is positive and a power of 2 */
	auto alignment = aLab->getAlignment();
	BUG_ON(alignment <= 0 || (alignment & (alignment - 1)) != 0);
	switch (aLab->getStorageDuration()) {
	case StorageDuration::SD_Automatic:
		return getAddrAllocator().allocAutomatic(aLab->getAllocSize(),
			alignment,
			aLab->getStorageType() == StorageType::ST_Durable,
			aLab->getAddressSpace() == AddressSpace::AS_Internal);
	case StorageDuration::SD_Heap:
		return getAddrAllocator().allocHeap(aLab->getAllocSize(),
			alignment,
			aLab->getStorageType() == StorageType::ST_Durable,
			aLab->getAddressSpace() == AddressSpace::AS_Internal);
	case StorageDuration::SD_Static: /* Cannot ask for fresh static addresses */
	default:
		BUG();
	}
	BUG();
	return SAddr();
}

int GenMCDriver::getFreshFd()
{
	int fd = fds.find_first_unset();

	/* If no available descriptor found, grow fds and try again */
	if (fd == -1) {
		fds.resize(2 * fds.size() + 1);
		return getFreshFd();
	}

	/* Otherwise, mark the file descriptor as used */
	markFdAsUsed(fd);
	return fd;
}

void GenMCDriver::markFdAsUsed(int fd)
{
	if (fd > fds.size())
		fds.resize(fd);
	fds.set(fd);
}

void GenMCDriver::resetThreadPrioritization()
{
	if (!userConf->LAPOR) {
		threadPrios.clear();
		return;
	}

	/*
	 * Check if there is some thread that did not manage to finish its
	 * critical section, and mark this execution as blocked
	 */
	const auto& g = getGraph();
	auto* EE = getEE();
	for (auto i = 0u; i < g.getNumThreads(); i++) {
		Event last = g.getLastThreadEvent(i);
		if (!g.getLastThreadUnmatchedLockLAPOR(last).isInitializer())
			EE->getThrById(i).block(BlockageType::LockNotRel);
	}

	/* Clear all prioritization */
	threadPrios.clear();
}

bool GenMCDriver::isLockWellFormedLAPOR() const
{
	if (!getConf()->LAPOR)
		return true;

	/*
	 * Check if there is some thread that did not manage to finish its
	 * critical section, and mark this execution as blocked
	 */
	const auto& g = getGraph();
	auto* EE = getEE();
	for (auto i = 0u; i < g.getNumThreads(); i++) {
		Event last = g.getLastThreadEvent(i);
		if (!g.getLastThreadUnmatchedLockLAPOR(last).isInitializer() &&
			std::none_of(EE->threads_begin(), EE->threads_end(), [](const llvm::Thread& thr) {
				return thr.getBlockageType() == BlockageType::Cons; }))
			return false;
	}
	return true;
}

void GenMCDriver::prioritizeThreads()
{
	if (!userConf->LAPOR)
		return;

	BUG();
	const auto& g = getGraph();

	/* Prioritize threads according to lock acquisitions */
	// threadPrios = g.getLbOrderingLAPOR();

	/* Remove threads that are executed completely */
	auto remIt = std::remove_if(threadPrios.begin(), threadPrios.end(), [&](Event e)
		{ return llvm::isa<ThreadFinishLabel>(g.getLastThreadLabel(e.thread)); });
	threadPrios.erase(remIt, threadPrios.end());
	return;
}

bool GenMCDriver::isSchedulable(int thread) const
{
	auto& thr = getEE()->getThrById(thread);
	auto* lab = getGraph().getLastThreadLabel(thread);
	return !thr.ECStack.empty() && !thr.isBlocked() && !lab->isTerminator();
}

bool GenMCDriver::schedulePrioritized()
{
	/* Return false if no thread is prioritized */
	if (threadPrios.empty())
		return false;

	const auto& g = getGraph();
	auto* EE = getEE();
	for (auto& e : threadPrios) {
		/* Skip unschedulable threads */
		if (!isSchedulable(e.thread))
			continue;

		/* Found a not-yet-complete thread; schedule it */
		EE->scheduleThread(e.thread);
		return true;
	}
	return false;
}

bool GenMCDriver::scheduleNextLTR()
{
	auto& g = getGraph();
	auto* EE = getEE();

	for (auto i = 0u; i < g.getNumThreads(); i++) {
		if (!isSchedulable(i))
			continue;

		/* Found a not-yet-complete thread; schedule it */
		EE->scheduleThread(i);
		return true;
	}

	/* No schedulable thread found */
	return false;
}

bool GenMCDriver::isNextThreadInstLoad(int tid)
{
	auto& I = getEE()->getThrById(tid).ECStack.back().CurInst;

	/* Overapproximate with function calls some of which might be modeled as loads */
	auto* ci = llvm::dyn_cast<CallInst>(I);
	return llvm::isa<llvm::LoadInst>(I) || llvm::isa<llvm::AtomicCmpXchgInst>(I) ||
		llvm::isa<llvm::AtomicRMWInst>(I) ||
		(ci && ci->getCalledFunction() && hasGlobalLoadSemantics(ci->getCalledFunction()->getName().str()));
}

bool GenMCDriver::scheduleNextWF()
{
	auto& g = getGraph();
	auto* EE = getEE();

	/* First, schedule based on the EG */
	for (auto i = 0u; i < g.getNumThreads(); i++) {
		if (!isSchedulable(i))
			continue;

		if (g.containsPos(Event(i, EE->getThrById(i).globalInstructions + 1))) {
			EE->scheduleThread(i);
			return true;
		}
	}

	/* Try and find a thread that satisfies the policy.
	 * Keep an LTR fallback option in case this fails */
	long fallback = -1;
	for (auto i = 0u; i < g.getNumThreads(); i++) {
		if (!isSchedulable(i))
			continue;

		if (fallback == -1)
			fallback = i;
		if (!isNextThreadInstLoad(i)) {
			EE->scheduleThread(getFirstSchedulableSymmetric(i));
			return true;
		}
	}

	/* Otherwise, try to schedule the fallback thread */
	if (fallback != -1) {
		EE->scheduleThread(getFirstSchedulableSymmetric(fallback));
		return true;
	}
	return false;
}

int GenMCDriver::getFirstSchedulableSymmetric(int tid)
{
	if (!getConf()->symmetryReduction)
		return tid;

	auto firstSched = tid;
	auto symm = getSymmPredTid(tid);
	while (symm != -1) {
		if (isSchedulable(symm))
			firstSched = symm;
		symm = getSymmPredTid(symm);
	}
	return firstSched;
}

bool GenMCDriver::scheduleNextWFR()
{
	auto& g = getGraph();
	auto* EE = getEE();

	/* First, schedule based on the EG */
	for (auto i = 0u; i < g.getNumThreads(); i++) {
		if (!isSchedulable(i))
			continue;

		if (g.containsPos(Event(i, EE->getThrById(i).globalInstructions + 1))) {
			EE->scheduleThread(i);
			return true;
		}
	}

	std::vector<int> nonwrites;
	std::vector<int> writes;
	for (auto i = 0u; i < g.getNumThreads(); i++) {
		if (!isSchedulable(i))
			continue;

		if (!isNextThreadInstLoad(i)) {
			writes.push_back(i);
		}
		else {
			nonwrites.push_back(i);
		}
	}

	std::vector<int>& selection = !writes.empty() ? writes : nonwrites;
	if (selection.empty())
		return false;

	MyDist dist(0, selection.size() - 1);
	auto candidate = selection[dist(rng)];
	EE->scheduleThread(getFirstSchedulableSymmetric(static_cast<int>(candidate)));
	return true;
}

bool GenMCDriver::scheduleNextRandom()
{
	auto& g = getGraph();
	auto* EE = getEE();

	/* Check if randomize scheduling is enabled and schedule some thread */
	MyDist dist(0, g.getNumThreads());
	auto random = dist(rng);
	for (auto j = 0u; j < g.getNumThreads(); j++) {
		auto i = (j + random) % g.getNumThreads();

		if (!isSchedulable(i))
			continue;

		/* Found a not-yet-complete thread; schedule it */
		EE->scheduleThread(getFirstSchedulableSymmetric(static_cast<int>(i)));
	}

	/* No schedulable thread found */
	return false;
}

void GenMCDriver::deprioritizeThread(const UnlockLabelLAPOR* uLab)
{
	/* Extra check to make sure the function is properly used */
	if (!userConf->LAPOR)
		return;

	auto& g = getGraph();

	auto delIt = threadPrios.end();
	for (auto it = threadPrios.begin(); it != threadPrios.end(); ++it) {
		auto* lLab = llvm::dyn_cast<LockLabelLAPOR>(g.getEventLabel(*it));
		BUG_ON(!lLab);

		if (lLab->getThread() == uLab->getThread() &&
			lLab->getLockAddr() == uLab->getLockAddr()) {
			delIt = it;
			break;
		}
	}

	if (delIt != threadPrios.end())
		threadPrios.erase(delIt);
	return;
}

void GenMCDriver::resetExplorationOptions()
{
	unmoot();
	setRescheduledRead(Event::getInit());
	resetThreadPrioritization();
}

void GenMCDriver::handleExecutionStart()
{
	const auto& g = getGraph();

	/* Set-up (optimize) the interpreter for the new exploration */
	for (auto i = 1u; i < g.getNumThreads(); i++) {

		/* Skip not-yet-created threads */
		BUG_ON(g.isThreadEmpty(i));

		auto* labFst = g.getFirstThreadLabel(i);
		auto parent = labFst->getParentCreate();

		/* Skip if parent create does not exist yet (or anymore) */
		if (!g.containsPos(parent) || !llvm::isa<ThreadCreateLabel>(g.getEventLabel(parent)))
			continue;

		/* Skip finished threads */
		auto* labLast = g.getLastThreadLabel(i);
		if (llvm::isa<ThreadFinishLabel>(labLast))
			continue;

		/* Skip the recovery thread, if it exists.
		 * It will be scheduled separately afterwards */
		if (i == g.getRecoveryRoutineId())
			continue;

		/* Otherwise, initialize ECStacks in interpreter */
		auto& thr = getEE()->getThrById(i);
		BUG_ON(!thr.ECStack.empty() || thr.isBlocked());
		thr.ECStack = thr.initEC;
	}

	/* Then, set up thread prioritization and interpreter's state */
	prioritizeThreads();
}

std::pair<std::vector<SVal>, Event>
GenMCDriver::extractValPrefix(Event pos)
{
	auto& g = getGraph();
	std::vector<SVal> vals;
	Event last;

	for (auto i = 0u; i < pos.index; i++) {
		auto* lab = g.getEventLabel(Event(pos.thread, i));
		if (auto* rLab = llvm::dyn_cast<ReadLabel>(lab)) {
			auto* drLab = llvm::dyn_cast<DskReadLabel>(rLab);
			vals.push_back(drLab ? getDskReadValue(drLab) : getReadValue(rLab));
			last = lab->getPos();
		}
		else if (auto* jLab = llvm::dyn_cast<ThreadJoinLabel>(lab)) {
			vals.push_back(getJoinValue(jLab));
			last = lab->getPos();
		}
		else if (auto* bLab = llvm::dyn_cast<ThreadStartLabel>(lab)) {
			vals.push_back(getStartValue(bLab));
			last = lab->getPos();
		}
		else if (auto* oLab = llvm::dyn_cast<OptionalLabel>(lab)) {
			vals.push_back(SVal(oLab->isExpanded()));
			last = lab->getPos();
		}
		else {
			BUG_ON(lab->hasValue());
		}
	}
	return { vals, last };
}

Event findNextLabelToAdd(const ExecutionGraph& g, Event pos)
{
	auto first = Event(pos.thread, 0);
	auto it = std::find_if(po_succ_begin(g, first), po_succ_end(g, first),
		[&](auto& lab) { return llvm::isa<EmptyLabel>(&lab); });
	return it == po_succ_end(g, first) ? g.getLastThreadEvent(pos.thread).next() : it->getPos();
}

bool GenMCDriver::tryOptimizeScheduling(Event pos)
{
	if (!getConf()->instructionCaching || inEstimationMode())
		return false;

	auto next = findNextLabelToAdd(getGraph(), pos);
	auto [vals, last] = extractValPrefix(next);
	auto* res = retrieveCachedSuccessors(pos.thread, vals);
	if (res == nullptr || res->empty() || res->back()->getIndex() < next.index)
		return false;

	for (auto& vlab : *res) {
		BUG_ON(vlab->hasStamp());

		DriverHandlerDispatcher dispatcher(this);
		dispatcher.visit(vlab);
		if (llvm::isa<BlockLabel>(getGraph().getLastThreadLabel(vlab->getThread())) ||
			isMoot() || getEE()->getCurThr().isBlocked() || isHalting())
			return true;
	}
	return true;
}

void GenMCDriver::checkHelpingCasAnnotation()
{
	/* If we were waiting for a helped CAS that did not appear, complain */
	if (std::any_of(getEE()->threads_begin(), getEE()->threads_end(),
		[](const llvm::Thread& thr) {
			return thr.getBlockageType() == BlockageType::HelpedCas;
		}))
		ERROR("Helped/Helping CAS annotation error! Does helped CAS always execute?\n");

	auto& g = getGraph();
	auto* EE = getEE();

	/* Next, we need to check whether there are any extraneous
	 * stores, not visible to the helped/helping CAS */
	auto hs = g.collectAllEvents([&](const EventLabel* lab) { return llvm::isa<HelpingCasLabel>(lab); });
	if (hs.empty())
		return;

	for (auto& h : hs) {
		auto* hLab = llvm::dyn_cast<HelpingCasLabel>(g.getEventLabel(h));
		BUG_ON(!hLab);

		/* Check that all stores that would make this helping
		 * CAS succeed are read by a helped CAS.
		 * We don't need to check the swap value of the helped CAS */
		if (std::any_of(store_begin(g, hLab->getAddr()), store_end(g, hLab->getAddr()),
			[&](auto& sLab) {
				return hLab->getExpected() == sLab.getVal() &&
					std::none_of(sLab.readers_begin(), sLab.readers_end(),
						[&](auto& rLab) {
							return llvm::isa<HelpedCasReadLabel>(&rLab);
						});
			}))
			ERROR("Helped/Helping CAS annotation error! "
				"Unordered store to helping CAS location!\n");

		/* Special case for the initializer (as above) */
		if (hLab->getAddr().isStatic() && hLab->getExpected() == EE->getLocInitVal(hLab->getAccess())) {
			auto rs = g.collectAllEvents([&](const EventLabel* lab) {
				auto* rLab = llvm::dyn_cast<ReadLabel>(lab);
				return rLab && rLab->getAddr() == hLab->getAddr();
				});
			if (std::none_of(rs.begin(), rs.end(), [&](const Event& r) {
				return llvm::isa<HelpedCasReadLabel>(g.getEventLabel(r));
				}))
				ERROR("Helped/Helping CAS annotation error! "
					"Unordered store to helping CAS location!\n");
		}
	}
	return;
}

bool GenMCDriver::isExecutionBlocked() const
{
	return std::any_of(getEE()->threads_begin(), getEE()->threads_end(),
		[this](const llvm::Thread& thr) {
			// FIXME: was thr.isBlocked()
			auto& g = getGraph();
			if (thr.id >= g.getNumThreads() || g.isThreadEmpty(thr.id)) // think rec
				return false;
			auto* bLab = llvm::dyn_cast<BlockLabel>(g.getLastThreadLabel(thr.id));
			return bLab || thr.isBlocked(); });
}

void GenMCDriver::updateStSpaceEstimation()
{
	/* Calculate current sample */
	auto& choices = getChoiceMap();
	auto sample = std::accumulate(choices.begin(), choices.end(), 1.0L,
		[](auto sum, auto& kv) { return sum *= kv.second.size(); });

	/* This is the (i+1)-th exploration */
	auto totalExplored = (long double)result.explored + result.exploredBlocked + 1L;

	/* As the estimation might stop dynamically, we can't just
	 * normalize over the max samples to avoid overflows. Instead,
	 * use Welford's online algorithm to calculate mean and
	 * variance. */
	auto prevM = result.estimationMean;
	auto prevV = result.estimationVariance;
	result.estimationMean += (sample - prevM) / totalExplored;
	result.estimationVariance += (sample - prevM) / totalExplored * (sample - result.estimationMean) -
		prevV / totalExplored;
	// if (result.explored + result.exploredBlocked <
	// getConf()->estimateRuns) { 	auto &g = getGraph();

	// 	MyDist dist(0, choices.size()-1);
	//         auto readIdx = dist(rng);
	// 	auto cIt = choices.begin();
	//         std::advance(cIt, readIdx);
	// 	auto readStamp = cIt->first;
	// 	auto *rLab = &*std::find_if(label_begin(g), label_end(g),
	// [&](auto &lab){ return lab.getStamp() == readStamp; });

	// 	MyDist dist2(0, choices[readStamp].size()-1);
	// 	auto storeIdx = dist2(rng);
	// 	auto *wLab = g.getEventLabel(choices[readStamp][storeIdx]);

	// 	if (rLab->getStamp() < wLab->getStamp()) {
	// 		llvm::dbgs() << "wlab is " << *wLab << "\n";
	// 		addToWorklist(wLab->getStamp(),
	// constructBackwardRevisit(llvm::dyn_cast<ReadLabel>(rLab),
	// llvm::dyn_cast<WriteLabel>(wLab))); 	} else {
	// 		addToWorklist(rLab->getStamp(),
	// std::make_unique<ReadForwardRevisit>(rLab->getPos(),
	// wLab->getPos()));
	// 	}
	// }
}



#ifdef DEBUG_LUAN
#define USE_RFSELECTOR

// 1

#ifndef USE_RFSELECTOR
void GenMCDriver::revisitCut() {
	auto& g = getGraph();
	// pick loads with multiple store options

	std::vector<RFPair> rf_options;

	auto isRMW = [&](auto lab, const ExecutionGraph& g) {return g.isRMWLoad(lab) || g.isRMWStore(lab); };
	for (auto i = 0u; i < g.getNumThreads(); i++) {
		for (auto j = 0u; j < g.getThreadSize(i); j++) {
			Event e(i, j);
			const EventLabel* lab = g.getEventLabel(e);
			BUG_ON(!lab);
			if (auto rLab = llvm::dyn_cast<ReadLabel>(lab); rLab && rLab->getRf()
				&& !rLab->isNotAtomic() && !(isRMW(rLab, g))) {
				Print("getting rf approx for:", *rLab);
				auto stores = getRfsApproximation(rLab); // assumes the read is latest in this thread
				// Print("rf selected: ", *rLab, "option: ", stores);
				// getchar();

				auto rmv = [&](const Event& e)
					{
						Print("try removing", e);
						return e == rLab->getRf()->getPos()
							;
					};
				stores.erase(std::remove_if(stores.begin(), stores.end(), rmv), stores.end());

				// push other rf alternatives for later selection
				if (stores.size() > 0) {
					for (auto& s : stores) {
						rf_options.push_back({ rLab, s });
					}
				}
			}
		}
	}


	if (rf_options.size() > 0) {
		MyDist dist(0, rf_options.size() - 1);
		auto rf_to_chg = rf_options[dist(estRng)];
		auto* rLab = rf_to_chg.first;
		BUG_ON(!rLab);
		auto store = rf_to_chg.second;

		Print("rf selected: ", *rLab, "option: ", store);

		// getchar();

		// get stamps and do forward/backward revist
		auto lstamp = rLab->getStamp();
		auto sLab = g.getWriteLabel(store);
		if (sLab) {
			Print("prefix view of the option:", getPrefixView(sLab));
		}

		BUG_ON((store != Event{ 0,0 } ? !sLab : false));
		// 
		auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };
		Print("load stamp: ", lstamp, ", store stamp: ", sstamp);
		// getchar();
		if (lstamp < sstamp) {
			// #ifdef FUZZ_BACKWARD
			// backward

			// getchar();
			const auto& gg = getGraph();
			auto br = constructBackwardRevisit(rLab, sLab);

			if (isMaximalExtension(*br)) { // not need				
				addToWorklist(sstamp, std::move(br));
			}
		}
		else {
			// forward
			addToWorklist(rLab->getStamp(), std::make_unique<ReadForwardRevisit>(rLab->getPos(), store, false));
			addToWorklist(rLab->getStamp(), std::make_unique<ReadForwardRevisit>(rLab->getPos(), store, false));	// push again
			Print("forward revisit");
			// revisit_flag = true;
		}


		// getchar();
	}


}


void GenMCDriver::minimalCut() {


	// copy paste...
	auto& g = getGraph();
	// pick loads with multiple store options
	using RFPair = std::pair<const ReadLabel* const, Event>;
	std::vector<RFPair> rf_options;
	for (auto i = 0u; i < g.getNumThreads(); i++) {
		for (auto j = 0u; j < g.getThreadSize(i); j++) {
			Event e(i, j);
			const EventLabel* lab = g.getEventLabel(e);
			BUG_ON(!lab);
			if (auto rLab = llvm::dyn_cast<ReadLabel>(lab)) {
				auto stores = getRfsApproximation(rLab);
				auto rmv = [&](const Event& e)
					{
						return e == rLab->getRf()->getPos()	// remove the current rf
							||
							getPrefixView(g.getEventLabel(e)).contains(rLab)
							;	// remove the current rf
					};
				stores.erase(std::remove_if(stores.begin(), stores.end(), rmv), stores.end());
				// push other rf alternatives for later selection
				if (stores.size() > 0) {
					for (auto& s : stores) {
						rf_options.push_back({ rLab, s });
					}
				}
			}
		}
	}


	if (rf_options.size() > 0) {
		MyDist dist(0, rf_options.size() - 1);
		auto rf_to_chg = rf_options[dist(estRng)];
		auto* rLab = rf_to_chg.first;
		BUG_ON(!rLab);
		auto store = rf_to_chg.second;

		Print("rf selected: ", *rLab, "option: ", store);

		// getchar();

		// get stamps and do forward/backward revist
		auto lstamp = rLab->getStamp();
		auto sLab = g.getWriteLabel(store);

		BUG_ON((store != Event{ 0,0 } ? !sLab : false));
		// 
		auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };
		Print("load stamp: ", lstamp, ", store stamp: ", sstamp);

		if (sLab) {
			auto porf_s = getPrefixView(sLab).clone();
			Print("porf_s:", *porf_s);
			auto porf_r = getPrefixView(rLab).clone();
			Print("porf_r:", *porf_r);
			// porf_r.update(porf_s);
			auto& updated = porf_s->update(*porf_r);
			Print("updated: ", updated);

			// g.changeRf(rLab->getPos(), store);


			auto og = g.getCopyUpTo(updated);
			auto maxStamp = og->getMaxStamp();
			Print("maxStamp of og", og->getMaxStamp());
			Print("maxStamp of g", g.getMaxStamp());
			auto m = createChoiceMapForCopy(*og);
			Print("copy up to updated:", *og);
			og->changeRf(rLab->getPos(), store);
			repairDanglingReads(*og);
			pushExecution({ std::move(og), LocalQueueT(), std::move(m) });
			addToWorklist(maxStamp, std::make_unique<RerunForwardRevisit>());
			// getchar();
		}


	}


}

void GenMCDriver::maximalCut() {
	// getchar();
	// copy paste...
	const auto& g = getGraph();
	// pick loads with multiple store options
	using RFPair = std::pair<const ReadLabel* const, Event>;
	std::vector<RFPair> rf_options;

	auto isRMW = [&](auto lab) {return g.isRMWLoad(lab) || g.isRMWStore(lab); };
	for (auto i = 0u; i < g.getNumThreads(); i++) {
		// Print("looking up thread", i);
		for (auto j = 0u; j < g.getThreadSize(i); j++) {
			Event e(i, j);
			const EventLabel* lab = g.getEventLabel(e);
			BUG_ON(!lab);
			if (auto rLab = llvm::dyn_cast<ReadLabel>(lab); rLab && rLab->getRf()
				&& !rLab->isNotAtomic()
				) {
				// BUG_ON(!rLab->getRf());
				auto stores = getRfsApproximation(rLab);
				// remove writes that are read by other rmw's
				auto isReadByOtherRMW = [&](const Event& e)
					{
						for (auto i = 0u; i < g.getNumThreads(); i++) {
							for (auto j = 0u; j < g.getThreadSize(i); j++) {
								Event e_(i, j);
								const EventLabel* lab = g.getEventLabel(e_);
								if (isRMW(lab) && isRMW(rLab)) {
									if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab)) {
										if (auto* ssLab = rrLab->getRf()) {
											if (ssLab->getPos() == e) return true;
										}
									}
								}
							}
						}
						return false;
					};

				auto rmv = [&](const Event& e)
					{
						auto lstamp = rLab->getStamp();
						auto sLab = g.getWriteLabel(e);
						auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };

						return e == rLab->getRf()->getPos()	// remove the current rf
							|| getPrefixView(g.getEventLabel(e)).contains(rLab)// e.thread == rLab->getThread()
							// || sstamp > lstamp
							|| isReadByOtherRMW(e)
							;
						;
					};
				stores.erase(std::remove_if(stores.begin(), stores.end(), rmv), stores.end());
				// push other rf alternatives for later selection
				if (stores.size() > 0) {
					for (auto& s : stores) {
						rf_options.push_back({ rLab, s });
					}
				}
			}
		}
	}
	Print("rf options prepared");


	if (rf_options.size() > 0) {
		MyDist dist(0, rf_options.size() - 1);
		auto rf_to_chg = rf_options[dist(estRng)];
		auto* rLab = rf_to_chg.first;
		BUG_ON(!rLab);
		auto store = rf_to_chg.second;

		Print("rf selected: ", *rLab, "option: ", store);

		if (isRMW(rLab)) {
			Print("rLab is rmw");
		}

		// getchar();

		// get stamps and do forward/backward revist
		auto lstamp = rLab->getStamp();
		auto sLab = g.getWriteLabel(store);

		BUG_ON((store != Event{ 0,0 } ? !sLab : false));
		// 
		// if (!sLab) return;
		auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };
		Print("load stamp: ", lstamp, ", store stamp: ", sstamp);

		if (sLab) {
			if (isRMW(sLab)) {
				Print("sLab is rmw");
			}
			auto porf_s = getPrefixView(sLab).clone();
			Print("porf_s:", *porf_s);
			auto porf_r = getPrefixView(rLab).clone();
			Print("porf_r:", *porf_r);
			auto& updated = porf_s->update(*porf_r);
			Print("minimal cut view: ", updated);
			auto toBeKept = std::make_unique<View>();




			for (auto i = 0u; i < g.getNumThreads(); i++) {
				for (auto j = 0u; j < g.getThreadSize(i); j++) {
					Event e(i, j);
					Print("checking", e);
					// if (
					// 	// i == rLab->getPos().thread
					// 	// ||
					// 	i == store.thread
					// 	) continue;

					const EventLabel* lab = g.getEventLabel(e);
					if (!getPrefixView(lab).contains(rLab)
						// && !getPrefixView(lab).contains(sLab)
						) {
						// if !read prefix(read)contains and !prefix(lab.rf).contains()
						// if !read prefix(join)contains and !prefix(join.finish).contains()
						bool updateFlag = true;
						// if (!Deps.empty())
						// 	Print("Deps:");
						// for (const auto& ee : Deps) {
						// 	Print("\t", ee);
						// }

						// for (const auto& ee : Deps) {
						// 	// Print("\t", ee);
						// 	if (auto eeLab = g.getEventLabel(ee))
						// 		if (getPrefixView(lab).contains(eeLab)) {
						// 			updateFlag = false;
						// 			Deps.push_back(e);
						// 		}

						// }


						if (auto jLab = llvm::dyn_cast<ThreadJoinLabel>(lab)) {
							if (auto* lLab = llvm::dyn_cast<ThreadFinishLabel>(g.getLastThreadLabel(jLab->getChildId()))) {
								if (getPrefixView(lLab).contains(rLab)) {
									updateFlag = false;
								}
								else {
									// toBeKept->updateIdx(jLab->getPos());
								}
							}
						}
						if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab)) {
							// Print("checking read", rrLab->getPos());
							if (auto* ssLab = rrLab->getRf()) {
								// Print("checking its write", ssLab->getPos());
								if (getPrefixView(ssLab).contains(rLab)) {
									updateFlag = false;
								}
								else {
									// toBeKept->updateIdx(ssLab->getPos());
								}
								if (ssLab->getPos().isBottom()) {
									Print(e, "reads from BOTTOM");
									updateFlag = false;
								}

								// if e is rmw and read is rmw
								// if e and read reads from the same write
								// false 
								// if (isRMW(rrLab) && isRMW(rLab)) {
								// 	if (ssLab->getPos() == sLab->getPos()) {
								// 		Print("two rmw rf same write! rrLab:", rrLab->getPos());
								// 		updateFlag = false;
								// 		// break;
								// 		Deps.push_back(e);
								// 	}
								// }
							}
							else {
								updateFlag = false;
							}
						}

						if (updateFlag) {
							Print("update idx", e);
							toBeKept->updateIdx(e);
						}
						else {
							Print("NOT update idx", e);
							// continue;

									// Deps.push_back(e);
							// break;
						}

					}
				}
			}


#if 0
			for (auto i = 0u; i < g.getNumThreads(); i++) {
				for (auto j = 0u; j < g.getThreadSize(i); j++) {
					Event e(i, j);
					const EventLabel* lab = g.getEventLabel(e);
					if (!getPrefixView(lab).contains(rLab)) {
						bool updateFlag = true;
						if (auto jLab = llvm::dyn_cast<ThreadJoinLabel>(lab)) {
							if (auto* lLab = llvm::dyn_cast<ThreadFinishLabel>(g.getLastThreadLabel(jLab->getChildId()))) {
								if (getPrefixView(lLab).contains(rLab)) {
									updateFlag = false;
								}
							}
						}
						if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab)) {
							if (auto* ssLab = rrLab->getRf()) {
								if (getPrefixView(ssLab).contains(rLab)) {
									updateFlag = false;
								}
							}
						}

						if (updateFlag) {
							toBeKept->updateIdx(e);
						}
					}
				}
			}
#endif



			toBeKept->updateIdx(rLab->getPos());
			Print("toBeKept before fix:", *toBeKept);

			// fix join
			// for (auto i = 0u; i < g.getNumThreads(); i++) {
			// 	for (auto j = 0u; j < g.getThreadSize(i); j++) {
			// 		Event e(i, j);
			// 		const EventLabel* lab = g.getEventLabel(e);
			// 		if (auto jLab = llvm::dyn_cast<ThreadJoinLabel>(lab)) {
			// 			Print("a join lable", e);
			// 			if (auto* lLab = llvm::dyn_cast<ThreadFinishLabel>(g.getLastThreadLabel(jLab->getChildId()))) {
			// 				if (toBeKept->contains(lLab->getPos())) {
			// 					Print("toBeKept contains last label", lLab->getPos());
			// 					toBeKept->updateIdx(e);
			// 				}
			// 			}
			// 		}
			// 	}
			// }





			// toBeKept->updateIdx(sLab->getPos());
			// if (store.thread != 0)
			// 	toBeKept->setMax(store);
			// if (auto orLab = rLab->getRf()) {
			// 	auto ostore = orLab->getPos();
			// 	if (ostore.thread != 0)
			// 		toBeKept->setMax({ ostore.thread, 1 });
			// }
			Print("maximal cut view:", *toBeKept);
			// #if 0


			{
				auto og = g.getCopyUpTo(*toBeKept);
				// auto og2 = g.getCopyUpTo(updated);
				// #endif
				Print("og:", *og);
				// Print("og2:", *og2);

				auto m = createChoiceMapForCopy(*og);
				auto maxStamp = og->getMaxStamp();
				Print("og's maxStamp = ", maxStamp, "g's maxStamp = ", g.getMaxStamp());
				og->changeRf(rLab->getPos(), store);
				repairDanglingReads(*og);
				// if (!og->violatesAtomicity(og->getWriteLabel(store))) {
				Print("copied graph", *og);
				pushExecution({ std::move(og), LocalQueueT(), std::move(m) });
				addToWorklist(maxStamp, std::make_unique<RerunForwardRevisit>());
				// }

			}
			{
				// auto porf_s = getPrefixView(sLab).clone();
				// auto porf_r = getPrefixView(rLab).clone();
				// auto& updated = porf_s->update(*porf_r);

				// auto og2 = g.getCopyUpTo(updated);
				// auto m = createChoiceMapForCopy(*og2);
				// auto maxStamp = og2->getMaxStamp();
				// og2->changeRf(rLab->getPos(), store);
				// repairDanglingReads(*og2);
				// pushExecution({ std::move(og2), LocalQueueT(), std::move(m) });
				// addToWorklist(maxStamp, std::make_unique<RerunForwardRevisit>());

			}
			// #else 

			// Print("getGraph()", getGraph());
		}


	}
}
#else


void GenMCDriver::revisitCut() {

	auto& g = getGraph();
	auto isRMW = [](auto lab, const ExecutionGraph& g) {return g.isRMWLoad(lab) || g.isRMWStore(lab); };
	auto pick = [isRMW](const EventLabel* lab, const ExecutionGraph& g) -> const ReadLabel*
		{
			auto rLab = llvm::dyn_cast<ReadLabel>(lab);
			if (rLab && rLab->getRf()
				&& !rLab->isNotAtomic()
				// && !(isRMW(rLab, g))
				) {
				return rLab;
			}
			return nullptr;
		};
	auto rmv = [](const ReadLabel* rLab, const ExecutionGraph& gg)
		{

			return [rLab, &gg](const Event& e) -> bool
				{
					Print("try removing", e);
					return e == rLab->getRf()->getPos();
				};
		};

	if (auto rf = selectAlternativeRF(g, pick, rmv)) {
		auto& rf_to_chg = *rf;
		auto* rLab = rf_to_chg.first;
		BUG_ON(!rLab);
		auto store = rf_to_chg.second;
		Print("rf selected: ", *rLab, "option: ", store);

		// getchar();

		// get stamps and do forward/backward revist
		auto lstamp = rLab->getStamp();
		auto sLab = g.getWriteLabel(store);
		if (sLab) {
			Print("prefix view of the option:", getPrefixView(sLab));
		}

		BUG_ON((store != Event{ 0,0 } ? !sLab : false));
		// 
		auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };
		Print("load stamp: ", lstamp, ", store stamp: ", sstamp);
		// getchar();
		if (lstamp < sstamp) {
			// #ifdef FUZZ_BACKWARD
			// backward

			// getchar();
			const auto& gg = getGraph();
			auto br = constructBackwardRevisit(rLab, sLab);

			if (isMaximalExtension(*br)) { // not need				
				addToWorklist(sstamp, std::move(br));
			}
		}
		else {
			// forward
			addToWorklist(rLab->getStamp(), std::make_unique<ReadForwardRevisit>(rLab->getPos(), store, false));
			addToWorklist(rLab->getStamp(), std::make_unique<ReadForwardRevisit>(rLab->getPos(), store, false));	// push again
			Print("forward revisit");
			// revisit_flag = true;
		}
	}
}



void GenMCDriver::do_minimalCut(RFPair rf_to_chg, const ExecutionGraph& g, bool push_to_gc) {

	auto* rLab = rf_to_chg.first;

	BUG_ON(!rLab);
	auto store = rf_to_chg.second;

	Print("rf selected: ", *rLab, "option: ", store);

	// getchar();

	// get stamps and do forward/backward revist
	auto lstamp = rLab->getStamp();
	auto sLab = g.getWriteLabel(store);

	BUG_ON((store != Event{ 0,0 } ? !sLab : false));
	// 
	auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };
	Print("load stamp: ", lstamp, ", store stamp: ", sstamp);

	if (sLab) {
		auto porf_s = getPrefixView(sLab).clone();
		Print("porf_s:", *porf_s);
		auto porf_r = getPrefixView(rLab).clone();
		Print("porf_r:", *porf_r);
		// porf_r.update(porf_s);
		auto& updated = porf_s->update(*porf_r);
		Print("updated: ", updated);

		// g.changeRf(rLab->getPos(), store);
		auto og = g.getCopyUpTo(updated);
		if (push_to_gc) {
			auto& gc = getGraphCounter();
			gc.addPrefix(std::move(og));
		}
		else {
			auto maxStamp = og->getMaxStamp();
			Print("maxStamp of og", og->getMaxStamp());
			Print("maxStamp of g", g.getMaxStamp());
			auto m = createChoiceMapForCopy(*og);
			Print("copy up to updated:", *og);
			og->changeRf(rLab->getPos(), store);
			repairDanglingReads(*og);
			pushExecution({ std::move(og), LocalQueueT(), std::move(m) });
			addToWorklist(maxStamp, std::make_unique<RerunForwardRevisit>());
		}


		// getchar();
	}
}

void GenMCDriver::minimalCut() {

	// copy paste...
	auto& g = getGraph();
	// pick loads with multiple store options
	auto pick = [](const EventLabel* lab, const ExecutionGraph& g) -> const ReadLabel*
		{
			return llvm::dyn_cast<ReadLabel>(lab);
		};
	auto rmv = [&](const ReadLabel* rLab, const ExecutionGraph& gg)
		{

			return [&gg, rLab, this](const Event& e) -> bool
				{
					Print("try removing", e);
					BUG_ON(!rLab->getRf());
					return e == rLab->getRf()->getPos()	// remove the current rf
						||
						getPrefixView(gg.getEventLabel(e)).contains(rLab)
						;
				};
		};
	if (auto rf = selectAlternativeRF(g, pick, rmv)) {
		do_minimalCut(*rf, g);
	}

}

void GenMCDriver::minimalCut_n(bool push_to_gc) {

	// copy paste...
	auto& g = getGraph();
	// pick loads with multiple store options
	auto pick = [](const EventLabel* lab, const ExecutionGraph& g) -> const ReadLabel*
		{
			return llvm::dyn_cast<ReadLabel>(lab);
		};
	auto rmv = [&](const ReadLabel* rLab, const ExecutionGraph& gg)
		{

			return [&gg, rLab, this](const Event& e) -> bool
				{
					Print("try removing", e);
					BUG_ON(!rLab->getRf());
					return e == rLab->getRf()->getPos()	// remove the current rf
						||
						getPrefixView(gg.getEventLabel(e)).contains(rLab)
						;
				};
		};
	if (auto rfs = selectAlternativeRF_n(g, pick, rmv, push_num); rfs.size()) {
		for (auto&& rf : rfs) do_minimalCut(rf, g, push_to_gc);
	}
	if (push_to_gc) {
		if (execStack.size() > 20) {
			execStack.pop_front();
		}
		auto& gc = getGraphCounter();
		if (!gc.is_sampling()) {
			auto ogg = gc.pickPrefix();
			if (ogg) {
				pushGraph(ogg);
			}
		}
	}

}


void GenMCDriver::do_maximalCut(RFPair rf_to_chg, const ExecutionGraph& g) {
	auto isRMW = [&](auto lab) {return g.isRMWLoad(lab) || g.isRMWStore(lab); };

	auto* rLab = rf_to_chg.first;
	BUG_ON(!rLab);
	auto store = rf_to_chg.second;

	Print("rf selected: ", *rLab, "option: ", store);

	if (isRMW(rLab)) {
		Print("rLab is rmw");
	}



	// get stamps and do forward/backward revist
	auto lstamp = rLab->getStamp();
	auto sLab = g.getWriteLabel(store);

	BUG_ON((store != Event{ 0,0 } ? !sLab : false));
	// 
	// if (!sLab) return;
	auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };
	Print("load stamp: ", lstamp, ", store stamp: ", sstamp);

	if (sLab) {
		if (isRMW(sLab)) {
			Print("sLab is rmw");
		}
		auto porf_s = getPrefixView(sLab).clone();
		Print("porf_s:", *porf_s);
		auto porf_r = getPrefixView(rLab).clone();
		Print("porf_r:", *porf_r);
		auto& updated = porf_s->update(*porf_r);
		Print("minimal cut view: ", updated);
		auto toBeKept = std::make_unique<View>();




		for (auto i = 0u; i < g.getNumThreads(); i++) {
			for (auto j = 0u; j < g.getThreadSize(i); j++) {
				Event e(i, j);
				Print("checking", e);

				const EventLabel* lab = g.getEventLabel(e);
				if (!getPrefixView(lab).contains(rLab)
					// && !getPrefixView(lab).contains(sLab)
					) {
					bool updateFlag = true;


					if (auto jLab = llvm::dyn_cast<ThreadJoinLabel>(lab)) {
						if (auto* lLab = llvm::dyn_cast<ThreadFinishLabel>(g.getLastThreadLabel(jLab->getChildId()))) {
							if (getPrefixView(lLab).contains(rLab)) {
								updateFlag = false;
							}
							else {
								// toBeKept->updateIdx(jLab->getPos());
							}
						}
					}
					if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab)) {
						// Print("checking read", rrLab->getPos());
						if (auto* ssLab = rrLab->getRf()) {
							// Print("checking its write", ssLab->getPos());
							if (getPrefixView(ssLab).contains(rLab)) {
								updateFlag = false;
							}
							else {
								// toBeKept->updateIdx(ssLab->getPos());
							}
							if (ssLab->getPos().isBottom()) {
								Print(e, "reads from BOTTOM");
								updateFlag = false;
							}
						}
						else {
							updateFlag = false;
						}
					}

					if (updateFlag) {
						Print("update idx", e);
						toBeKept->updateIdx(e);
					}
					else {
						Print("NOT update idx", e);
					}

				}
			}
		}

		toBeKept->updateIdx(rLab->getPos());
		Print("toBeKept before fix:", *toBeKept);

		Print("maximal cut view:", *toBeKept);


		// #ifdef POWER_SCHED
		// 			auto og = g.getCopyUpTo(*toBeKept);
		// 			og->changeRf(rLab->getPos(), store);
		// 			repairDanglingReads(*og);

		// 			auto& gc = getGraphCounter();
		// 			gc.addPrefix(std::move(og));

		// 			if (!gc.is_sampling()) {
		// 				auto ogg = gc.pickPrefix();
		// 				if (ogg) {
		// 					pushGraph(ogg);
		// 				}
		// 			}
		// #else
		{
			auto og = g.getCopyUpTo(*toBeKept);
			Print("og:", *og);
			auto m = createChoiceMapForCopy(*og);
			auto maxStamp = og->getMaxStamp();
			Print("og's maxStamp = ", maxStamp, "g's maxStamp = ", g.getMaxStamp());
			og->changeRf(rLab->getPos(), store);
			repairDanglingReads(*og);

			Print("copied graph", *og);
			pushExecution({ std::move(og), LocalQueueT(), std::move(m) });
			addToWorklist(maxStamp, std::make_unique<RerunForwardRevisit>());
		}
		// #endif	// POWER_SCHED

	}
}

void GenMCDriver::maximalCut() {

	const auto& g = getGraph();
	// pick loads with multiple store options
	using RFPair = std::pair<const ReadLabel* const, Event>;
	std::vector<RFPair> rf_options;

	auto pick = [](const EventLabel* lab, const ExecutionGraph& g) -> const ReadLabel*
		{
			auto rLab = llvm::dyn_cast<ReadLabel>(lab);
			return rLab && rLab->getRf()
				&& !rLab->isNotAtomic() ? rLab : nullptr;
		};



	auto isRMW = [&](auto lab) {return g.isRMWLoad(lab) || g.isRMWStore(lab); };
	auto isReadByOtherRMW = [isRMW](const Event& e, const ReadLabel* rLab, const ExecutionGraph& g)
		{
			for (auto i = 0u; i < g.getNumThreads(); i++) {
				for (auto j = 0u; j < g.getThreadSize(i); j++) {
					Event e_(i, j);
					const EventLabel* lab = g.getEventLabel(e_);
					if (isRMW(lab) && isRMW(rLab)) {
						if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab)) {
							if (auto* ssLab = rrLab->getRf()) {
								if (ssLab->getPos() == e) return true;
							}
						}
					}
				}
			}
			return false;
		};
	auto rmv = [this, isReadByOtherRMW](const ReadLabel* rLab, const ExecutionGraph& gg)
		{
			auto& tmp = isReadByOtherRMW;
			return [&gg, rLab, this, tmp](const Event& e) -> bool
				{
					Print("try removing", e);
					auto lstamp = rLab->getStamp();
					auto sLab = gg.getWriteLabel(e);
					auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };

					return e == rLab->getRf()->getPos()	// remove the current rf
						|| getPrefixView(gg.getEventLabel(e)).contains(rLab)// e.thread == rLab->getThread()
						// || sstamp > lstamp
						|| tmp(e, rLab, gg)
						;
					;
				};
		};

	if (auto rf = selectAlternativeRF(g, pick, rmv)) {
		do_maximalCut(*rf, g);
	}


}




void GenMCDriver::powerSched() {

	const auto& g = getGraph();
	// pick loads with multiple store options
	using RFPair = std::pair<const ReadLabel* const, Event>;
	std::vector<RFPair> rf_options;

	auto pick = [](const EventLabel* lab, const ExecutionGraph& g) -> const ReadLabel*
		{
			auto rLab = llvm::dyn_cast<ReadLabel>(lab);
			return rLab && rLab->getRf()
				&& !rLab->isNotAtomic() ? rLab : nullptr;
		};



	auto isRMW = [&](auto lab) {return g.isRMWLoad(lab) || g.isRMWStore(lab); };
	auto isReadByOtherRMW = [isRMW](const Event& e, const ReadLabel* rLab, const ExecutionGraph& g)
		{
			for (auto i = 0u; i < g.getNumThreads(); i++) {
				for (auto j = 0u; j < g.getThreadSize(i); j++) {
					Event e_(i, j);
					const EventLabel* lab = g.getEventLabel(e_);
					if (isRMW(lab) && isRMW(rLab)) {
						if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab)) {
							if (auto* ssLab = rrLab->getRf()) {
								if (ssLab->getPos() == e) return true;
							}
						}
					}
				}
			}
			return false;
		};
	auto rmv = [this, isReadByOtherRMW](const ReadLabel* rLab, const ExecutionGraph& gg)
		{
			auto& tmp = isReadByOtherRMW;
			return [&gg, rLab, this, tmp](const Event& e) -> bool
				{
					Print("try removing", e);
					auto lstamp = rLab->getStamp();
					auto sLab = gg.getWriteLabel(e);
					auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };

					return e == rLab->getRf()->getPos()	// remove the current rf
						|| getPrefixView(gg.getEventLabel(e)).contains(rLab)// e.thread == rLab->getThread()
						// || sstamp > lstamp
						|| tmp(e, rLab, gg)
						;
					;
				};
		};

	if (auto rf = selectAlternativeRF(g, pick, rmv)) {
		auto rf_to_chg = *rf;
		auto* rLab = rf_to_chg.first;
		BUG_ON(!rLab);
		auto store = rf_to_chg.second;

		Print("rf selected: ", *rLab, "option: ", store);

		if (isRMW(rLab)) {
			Print("rLab is rmw");
		}



		// get stamps and do forward/backward revist
		auto lstamp = rLab->getStamp();
		auto sLab = g.getWriteLabel(store);

		BUG_ON((store != Event{ 0,0 } ? !sLab : false));
		// 
		// if (!sLab) return;
		auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };
		Print("load stamp: ", lstamp, ", store stamp: ", sstamp);

		if (sLab) {
			if (isRMW(sLab)) {
				Print("sLab is rmw");
			}
			auto porf_s = getPrefixView(sLab).clone();
			Print("porf_s:", *porf_s);
			auto porf_r = getPrefixView(rLab).clone();
			Print("porf_r:", *porf_r);
			auto& updated = porf_s->update(*porf_r);
			Print("minimal cut view: ", updated);
			auto toBeKept = std::make_unique<View>();




			for (auto i = 0u; i < g.getNumThreads(); i++) {
				for (auto j = 0u; j < g.getThreadSize(i); j++) {
					Event e(i, j);
					Print("checking", e);

					const EventLabel* lab = g.getEventLabel(e);
					if (!getPrefixView(lab).contains(rLab)
						// && !getPrefixView(lab).contains(sLab)
						) {
						bool updateFlag = true;


						if (auto jLab = llvm::dyn_cast<ThreadJoinLabel>(lab)) {
							if (auto* lLab = llvm::dyn_cast<ThreadFinishLabel>(g.getLastThreadLabel(jLab->getChildId()))) {
								if (getPrefixView(lLab).contains(rLab)) {
									updateFlag = false;
								}
								else {
									// toBeKept->updateIdx(jLab->getPos());
								}
							}
						}
						if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab)) {
							// Print("checking read", rrLab->getPos());
							if (auto* ssLab = rrLab->getRf()) {
								// Print("checking its write", ssLab->getPos());
								if (getPrefixView(ssLab).contains(rLab)) {
									updateFlag = false;
								}
								else {
									// toBeKept->updateIdx(ssLab->getPos());
								}
								if (ssLab->getPos().isBottom()) {
									Print(e, "reads from BOTTOM");
									updateFlag = false;
								}
							}
							else {
								updateFlag = false;
							}
						}

						if (updateFlag) {
							Print("update idx", e);
							toBeKept->updateIdx(e);
						}
						else {
							Print("NOT update idx", e);
						}

					}
				}
			}

			toBeKept->updateIdx(rLab->getPos());
			Print("toBeKept before fix:", *toBeKept);

			Print("maximal cut view:", *toBeKept);



			auto og = g.getCopyUpTo(*toBeKept);
			og->changeRf(rLab->getPos(), store);
			repairDanglingReads(*og);

			auto& gc = getGraphCounter();
			gc.addPrefix(std::move(og));

			if (!gc.is_sampling()) {
				auto ogg = gc.pickPrefix();
				if (ogg) {
					pushGraph(ogg);
				}
			}


		}
	}


}

template<typename T, typename U>
void GenMCDriver::do_maximalCut2(RFPair rf_to_chg, const ExecutionGraph& g, T&& isRMW, U&& isReadByOtherRMW) {

	auto* rLab = rf_to_chg.first;
	BUG_ON(!rLab);
	auto store = rf_to_chg.second;

	Print("rf selected: ", *rLab, "option: ", store);

	if (isRMW(rLab)) {
		Print("rLab is rmw");
	}



	// get stamps and do forward/backward revist
	auto lstamp = rLab->getStamp();
	auto sLab = g.getWriteLabel(store);

	BUG_ON((store != Event{ 0,0 } ? !sLab : false));
	// 
	// if (!sLab) return;
	auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };
	Print("load stamp: ", lstamp, ", store stamp: ", sstamp);

	if (sLab) {
		if (isRMW(sLab)) {
			Print("sLab is rmw");
		}

		auto toBeKept = std::make_unique<View>();


		// std::optional<Event> otherRMW;
		// for (auto i = 0u; i < g.getNumThreads(); i++) {
		// 	for (auto j = 0u; j < g.getThreadSize(i); j++) {
		// 		Event e(i, j);
		// 		const EventLabel* lab = g.getEventLabel(e);
		// 		if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab); rrLab && isRMW(rrLab)) {
		// 			if (auto* ssLab = rrLab->getRf(); ssLab && ssLab->getPos() == store) {
		// 				// BUG_ON(otherRMW);
		// 				otherRMW = rrLab->getPos();;
		// 			}
		// 		}
		// 	}
		// }

		auto isPorfSus = [&](const EventLabel* lab)
			{
				auto res = getPrefixView(lab).contains(rLab);
				// if (otherRMW) {
				// 	auto o = *otherRMW;
				// 	res &= getPrefixView(lab).contains(o);
				// }
				return res;
			};

		for (auto i = 0u; i < g.getNumThreads(); i++) {
			for (auto j = 0u; j < g.getThreadSize(i); j++) {
				Event e(i, j);
				Print("checking", e);

				const EventLabel* lab = g.getEventLabel(e);
				if (!isPorfSus(lab)
					// && !getPrefixView(lab).contains(sLab)
					) {
					bool updateFlag = true;


					if (auto jLab = llvm::dyn_cast<ThreadJoinLabel>(lab)) {
						if (auto* lLab = llvm::dyn_cast<ThreadFinishLabel>(g.getLastThreadLabel(jLab->getChildId()))) {
							if (isPorfSus(lLab)) {
								updateFlag = false;
							}
							else {
								// toBeKept->updateIdx(jLab->getPos());
							}
						}
					}
					if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab)) {
						// Print("checking read", rrLab->getPos());
						if (auto* ssLab = rrLab->getRf()) {
							// Print("checking its write", ssLab->getPos());
							if (isPorfSus(ssLab)) {
								updateFlag = false;
							}
							else {
								// toBeKept->updateIdx(ssLab->getPos());
							}
							if (ssLab->getPos().isBottom()) {
								Print(e, "reads from BOTTOM");
								updateFlag = false;
							}
						}
						else {
							updateFlag = false;
						}
					}

					// if (otherRMW) {
					// 	auto o = *otherRMW;
					// 	if (e == o) {
					// 		updateFlag = false;
					// 	}
					// }

					if (updateFlag) {
						Print("update idx", e);
						toBeKept->updateIdx(e);
					}
					else {
						Print("NOT update idx", e);
					}

				}
			}
		}

		toBeKept->updateIdx(rLab->getPos());
		Print("toBeKept before fix:", *toBeKept);

		Print("maximal cut view:", *toBeKept);


#ifdef POWER_SCHED
		auto og = g.getCopyUpTo(*toBeKept);
		og->changeRf(rLab->getPos(), store);
		repairDanglingReads(*og);

		auto& gc = getGraphCounter();
		gc.addPrefix(std::move(og));

		if (!gc.is_sampling()) {
			auto ogg = gc.pickPrefix();
			if (ogg) {
				pushGraph(ogg);
			}
		}
#else
		{
			auto og = g.getCopyUpTo(*toBeKept);
			Print("og:", *og);
			auto m = createChoiceMapForCopy(*og);
			auto maxStamp = og->getMaxStamp();
			Print("og's maxStamp = ", maxStamp, "g's maxStamp = ", g.getMaxStamp());
			if (!isReadByOtherRMW(store, rLab, g)) {
				og->changeRf(rLab->getPos(), store);
				repairDanglingReads(*og);
			}


			Print("copied graph", *og);
			pushExecution({ std::move(og), LocalQueueT(), std::move(m) });
			addToWorklist(maxStamp, std::make_unique<RerunForwardRevisit>());
		}
#endif	// POWER_SCHED

	}
}


template<bool PushOnce>
void GenMCDriver::maximalCut2() {

	const auto& g = getGraph();
	// pick loads with multiple store options
	using RFPair = std::pair<const ReadLabel* const, Event>;
	std::vector<RFPair> rf_options;

	auto pick = [](const EventLabel* lab, const ExecutionGraph& g) -> const ReadLabel*
		{
			auto rLab = llvm::dyn_cast<ReadLabel>(lab);
			return rLab && rLab->getRf()
				&& !rLab->isNotAtomic() ? rLab : nullptr;
		};



	auto isRMW = [&](auto lab) {return g.isRMWLoad(lab) || g.isRMWStore(lab); };
	auto isReadByOtherRMW = [isRMW](const Event& e, const ReadLabel* rLab, const ExecutionGraph& g)
		{
			for (auto i = 0u; i < g.getNumThreads(); i++) {
				for (auto j = 0u; j < g.getThreadSize(i); j++) {
					Event e_(i, j);
					const EventLabel* lab = g.getEventLabel(e_);
					if (isRMW(lab) && isRMW(rLab)) {
						if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab)) {
							if (auto* ssLab = rrLab->getRf()) {
								if (ssLab->getPos() == e) return true;
							}
						}
					}
				}
			}
			return false;
		};
	auto rmv = [this, isReadByOtherRMW](const ReadLabel* rLab, const ExecutionGraph& gg)
		{
			auto& tmp = isReadByOtherRMW;
			return [&gg, rLab, this, tmp](const Event& e) -> bool
				{
					Print("try removing", e);
					auto lstamp = rLab->getStamp();
					auto sLab = gg.getWriteLabel(e);
					auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };

					return e == rLab->getRf()->getPos()	// remove the current rf
						|| getPrefixView(gg.getEventLabel(e)).contains(rLab)// e.thread == rLab->getThread()
						// || sstamp > lstamp
						// || tmp(e, rLab, gg)
						;
					;
				};
		};

	if constexpr (PushOnce) {
		if (auto rf = selectAlternativeRF(g, pick, rmv)) {
			do_maximalCut2(*rf, g, isRMW, isReadByOtherRMW);
		}
	}
	else {
		if (auto rfs = selectAlternativeRF_n(g, pick, rmv, push_num); rfs.size()) {
			for (auto&& rf : rfs) do_maximalCut2(rf, g, isRMW, isReadByOtherRMW);
		}
	}




}


void GenMCDriver::maximalCut2_n() {}




#endif	// USE_RFSELECTOR



void GenMCDriver::maximalCutRMW() {
	// getchar();
	// copy paste...
	const auto& g = getGraph();
	// pick loads with multiple store options
	using RFPair = std::pair<const ReadLabel* const, Event>;
	std::vector<RFPair> rf_options;

	auto isRMW = [&](auto lab, const ExecutionGraph& g) {return g.isRMWLoad(lab) || g.isRMWStore(lab); };


	for (auto i = 0u; i < g.getNumThreads(); i++) {
		// Print("looking up thread", i);
		for (auto j = 0u; j < g.getThreadSize(i); j++) {
			Event e(i, j);
			const EventLabel* lab = g.getEventLabel(e);
			BUG_ON(!lab);
			if (auto rLab = llvm::dyn_cast<ReadLabel>(lab); rLab && rLab->getRf() && !rLab->isNotAtomic()) {
				// BUG_ON(!rLab->getRf());
				// Print("getting rf for:", e);
				auto stores = getRfsApproximation(rLab);

				auto rmv = [&](const Event& e)
					{
						auto lstamp = rLab->getStamp();
						auto sLab = g.getWriteLabel(e);
						auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };
						// Print("prefix of", e, ":", getPrefixView(g.getEventLabel(e)));

						return e == rLab->getRf()->getPos()	// remove the current rf
							|| getPrefixView(g.getEventLabel(e)).contains(rLab)// e.thread == rLab->getThread()

							;
						;
					};
				stores.erase(std::remove_if(stores.begin(), stores.end(), rmv), stores.end());
				// push other rf alternatives for later selection
				if (stores.size() > 0) {
					// Print("read: ", *rLab);
					for (auto& s : stores) {
						// Print("\toption:", s);
						rf_options.push_back({ rLab, s });
					}
					// getchar();
				}

			}
		}
	}
	Print("rf options prepared");


	if (rf_options.size() > 0) {
		MyDist dist(0, rf_options.size() - 1);
		auto rf_to_chg = rf_options[dist(estRng)];
		auto* rLab = rf_to_chg.first;
		BUG_ON(!rLab);
		auto store = rf_to_chg.second;

		Print("rf selected: ", *rLab, "option: ", store);

		if (isRMW(rLab, g)) {
			Print("rLab", *rLab, "is rmw");
		}

		// getchar();

		// get stamps and do forward/backward revist
		auto lstamp = rLab->getStamp();
		auto sLab = g.getWriteLabel(store);

		BUG_ON((store != Event{ 0,0 } ? !sLab : false));
		// 
		// if (!sLab) return;
		auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };
		Print("load stamp: ", lstamp, ", store stamp: ", sstamp);

		if (sLab) {


			// remove porf successors of rLab, OTHER than rLab itself
			auto shouldUpdate = [&](const EventLabel* lab, const EventLabel* rLab, const ExecutionGraph& g)
				{
					BUG_ON(!lab);
					BUG_ON(!rLab);
					bool updateFlag = false;
					// Print("getPrefixView(lab):", getPrefixView(lab));
					// Print("rLab:", rLab->getPos());
					if (!getPrefixView(lab).contains(rLab)) {
						updateFlag = true;
						if (auto jLab = llvm::dyn_cast<ThreadJoinLabel>(lab)) {
							if (auto* lLab = llvm::dyn_cast<ThreadFinishLabel>(g.getLastThreadLabel(jLab->getChildId()))) {
								if (getPrefixView(lLab).contains(rLab)) {
									updateFlag = false;
								}
							}
						}
						if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab)) {
							// Print("checking read", rrLab->getPos());
							if (auto* ssLab = rrLab->getRf()) {
								// Print("checking its write", ssLab->getPos());
								if (getPrefixView(ssLab).contains(rLab)) {
									updateFlag = false;
								}

								if (ssLab->getPos().isBottom()) {
									updateFlag = false;
								}
							}
							else {
								updateFlag = false;
							}
						}
					}
					return updateFlag;
				};


			auto includeNonPoRfSuccessor = [&](const EventLabel* rLab, const ExecutionGraph& g, std::unique_ptr<View>& v)
				{
					// Print("including for", rLab->getPos());
					for (auto i = 0u; i < g.getNumThreads(); i++) {
						for (auto j = 0u; j < g.getThreadSize(i); j++) {
							Event e(i, j);
							// Print("try including", e);
							const EventLabel* lab = g.getEventLabel(e);
							if (shouldUpdate(lab, rLab, g)) {
								// Print("should update");
								v->updateIdx(e);
								// Print("updated:", *v);
							}
						}
					}
				};
			//////////////////////////////////////////////////////////////////////
						// sLab: the target write 
						// rLab: the mutated read		--> event: er
						// 	   : the write after the mutated read --> event e2

			auto toBeKept = std::make_unique<View>();
			includeNonPoRfSuccessor(rLab, g, toBeKept);

			toBeKept->updateIdx(rLab->getPos());
			Print("STEP1: after remove porf succ of rLab, toBeKept:", *toBeKept);

			// check if other rmw reads from the same store that rLab reads from
			std::optional<std::pair<Event, Event> > reorderedRMW;
			if (auto er = rLab->getPos(); isRMW(rLab, g)) {
				auto e2 = er.next();	// the rmw write after the current rmw read
				Print("ther write after rLab(rmw)", e2);
				if (g.getWriteLabel(e2)) {
					toBeKept->updateIdx(e2);
				}
				Print("STEP1: included e2(after rLab)", *toBeKept);
				// getchar();
				// if other rmw reads from store, let it read from e2
				for (auto i = 0u; i < g.getNumThreads(); i++) {
					for (auto j = 0u; j < g.getThreadSize(i); j++) {
						Event e(i, j);
						const EventLabel* lab = g.getEventLabel(e);
						if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab); rrLab && isRMW(rrLab, g)) {
							if (auto* ssLab = rrLab->getRf(); ssLab && ssLab->getPos() == store) {
								auto reord = rrLab->getPos();
								Print("STEP1: found rmw to be reordered:", reord);
								BUG_ON(reorderedRMW);
								reorderedRMW = { reord, e2 };
							}
						}
					}
				}

			}
			// not mutated so far

			Print("STEP1: cut view:", *toBeKept);
			auto og = g.getCopyUpTo(*toBeKept);
			Print("STEP1: copied og:", *og);
			// getchar();
			// change rf first, then check

			og->changeRf(rLab->getPos(), store);
			Print("STEP1: change rf for rLab", *og);
			// getchar();
			// filling in hole
			{
				rLab = og->getReadLabel(rLab->getPos());	// label of copied graph
				BUG_ON(!rLab);
				Print("STEP1: since rLab is rmw, repairt the write after it");
				const auto w = rLab->getPos().next();
				auto wLab = og->getWriteLabel(w);
				BUG_ON(!wLab);

				og->setEventLabel(w, og->createHoleLabel(w));
				og->removeStoreFromCO(wLab);
				Print("filled in hole", *og);
				// getchar();
			}


			// first complete the write part of the mutated rmw
			auto v = std::make_unique<View>();
			if (auto* rsLab = completeRevisitedRMW(og->getReadLabel(rLab->getPos()), *og)) {
				Print("STEP1: complete the write part of the mutated rmw", *og);
				// getchar();
				if (reorderedRMW) {
					Print("reordering rmw");
					auto e = (*reorderedRMW).first;
					auto* rrLab = og->getReadLabel(e);
					BUG_ON(!rrLab);
					// compute a second time BEFORE change rf for the second time
					auto lab = og->getEventLabel(e.next());
					// auto rLab = llvm::dyn_cast<ReadLabel>(lab);
					// auto rsLab = llvm::dyn_cast<ReadLabel>(og->getEventLabel(s));
					// rsLab->reset();	// ?????????????????
					// BUG_ON(!rLab);
					BUG_ON(!lab);
					og->changeRf(e, rsLab->getPos());
					includeNonPoRfSuccessor(lab, *og, v);
					v->updateIdx(rsLab->getPos());
					v->updateIdx(sLab->getPos());

					// 
					// Print("changed rf, before fixing write", *og);
					const auto w = e.next();
					auto wLab = og->getWriteLabel(w);
					BUG_ON(!wLab);
					og->removeStoreFromCO(wLab);

					og->setEventLabel(w, og->createHoleLabel(w));
					auto ssLab = completeRevisitedRMW(rrLab, *og);
					// repairDanglingReads(*og);
					Print("reorder done", *og);
				}
			}

			repairDanglingReads(*og);





			if (reorderedRMW) {
				Print("cutting for reordered, og = ", *og);
				auto [r, s] = *reorderedRMW;
				Print("r = ", r, ", s =", s, ", r.next() =", r.next());


				v->updateIdx(r.next());
				v->updateIdx(s);
				Print("v = ", *v);
				// update reads:
				// for (auto i = 0u; i < og->getNumThreads(); i++) {
				// 	for (auto j = 0u; j < og->getThreadSize(i); j++) {
				// 		Event e(i, j);
				// 		const EventLabel* lab = g.getEventLabel(e);
				// 		if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab)) {
				// 			if (auto* ssLab = rrLab->getRf()) {
				// 				v->updateIdx(ssLab->getPos());
				// 			}
				// 		}
				// 	}
				// }
				// getchar();
				auto og2 = og->getCopyUpTo(*v);
				Print("og2:", *og2);
				// getchar();
				pushGraph(og2);


				// og->cutToStamp(std::max(og->getEventLabel(r.next())->getStamp(), og->getEventLabel(s)->getStamp()));
				// pushGraph(og);



				// og->changeRf(r, s);
			}
			else {
				pushGraph(og);

			}
		}
	}
}


void GenMCDriver::do_maximalCutRMW2(RFPair rf_to_chg, const ExecutionGraph& g) {

	auto isRMW = [&](auto lab, const ExecutionGraph& g)
		{
			auto is = g.isRMWLoad(lab) || g.isRMWStore(lab);
			if (llvm::isa<CasReadLabel>(lab) || llvm::isa<CasWriteLabel>(lab))
				is = false;
			return is;
		};

	auto* rLab = rf_to_chg.first;
	BUG_ON(!rLab);
	auto store = rf_to_chg.second;

	Print("rf selected: ", *rLab, "option: ", store);


	auto sLab = g.getWriteLabel(store);

	BUG_ON((store != Event{ 0,0 } ? !sLab : false));

	//--------------notation----------------//
	//	    	 [sLab, store]
	//		/					\ 
	//	[rLab, load1]  		[rLab2, load2]
	//					 /
	//  [rsLab,store1]  /	[rsLab2,store2]


	if (sLab) {

		// remove porf successors of rLab, OTHER than rLab itself
		auto shouldUpdate = [&](const EventLabel* lab, const EventLabel* rLab, const ExecutionGraph& g)
			{
				BUG_ON(!lab);
				BUG_ON(!rLab);
				bool updateFlag = false;
				// Print(lab->getPos(), "getPrefixView(lab):", getPrefixView(lab));
				// Print("rLab:", rLab->getPos());
				if (auto rrlab = llvm::dyn_cast<ReadLabel>(lab); rrlab && rrlab->getRf()) {
					// if rLab is a write lab (naming's fault)
					if (rrlab->getRf()->getPos() == rLab->getPos() && llvm::dyn_cast<WriteLabel>(rLab)) {
						// updateFlag = false;
					}
				}

				if (!getPrefixView(lab).contains(rLab)) {
					updateFlag = true;
					if (auto jLab = llvm::dyn_cast<ThreadJoinLabel>(lab)) {
						if (auto* lLab = llvm::dyn_cast<ThreadFinishLabel>(g.getLastThreadLabel(jLab->getChildId()))) {
							BUG_ON(!rLab);
							if (getPrefixView(lLab).contains(rLab)) {
								updateFlag = false;
							}
						}
					}
					if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab)) {
						Print("checking read", rrLab->getPos());
						if (auto* ssLab = rrLab->getRf()) {
							Print("checking its write", ssLab->getPos());
							if (getPrefixView(ssLab).contains(rLab)) {
								Print("contains. flag = false");
								updateFlag = false;
							}

							if (ssLab->getPos().isBottom()) {
								updateFlag = false;
							}
						}
						else {
							updateFlag = false;
						}
					}
				}
				return updateFlag;
			};


		auto includeNonPoRfSuccessor = [&](const EventLabel* rLab, const ExecutionGraph& g, std::unique_ptr<View>& v)
			{
				// Print("including for", rLab->getPos());
				for (auto i = 0u; i < g.getNumThreads(); i++) {
					for (auto j = 0u; j < g.getThreadSize(i); j++) {
						Event e(i, j);
						// Print("try including", e);
						const EventLabel* lab = g.getEventLabel(e);
						// for read part of rmw, check the write part of it
						if (g.isRMWLoad(lab)) continue;

						if (shouldUpdate(lab, rLab, g)) {
							// Print("should update");
							v->updateIdx(e);
							// Print("updated:", *v);
						}

					}
				}
			};

		auto updateAllViews = [&]()
			{
				Print("updating all views");
				auto& g = getGraph();
				for (auto i = 0u; i < g.getNumThreads(); i++) {
					for (auto j = 0u; j < g.getThreadSize(i); j++) {
						Event e(i, j);

						const EventLabel* lab = g.getEventLabel(e);
						// Print("	try updating", e, *lab);
						if (!llvm::isa<EmptyLabel>(lab))
							updateLabelViews((EventLabel*)lab);
					}
				}
			};
		//////////////////////////////////////////////////////////////////////


		printPrefixViewsOnStack("before doing anything");
		updateAllViews();
		printPrefixViewsOnStack("after updateAllViews");
		auto toBeKept = std::make_unique<View>();
		Print("computing view for rLab");

		includeNonPoRfSuccessor(rLab, g, toBeKept);

		toBeKept->updateIdx(rLab->getPos());
		Print("after remove porf succ of rLab, toBeKept:", *toBeKept);
		// getchar();

		// check if other rmw reads from the same store that rLab reads from
		std::optional<std::pair<Event, Event> > reorderedRMW;
		if (auto load1 = rLab->getPos(); isRMW(rLab, g)) {
			auto store1 = load1.next();	// the rmw write after the current rmw read

			if (g.getWriteLabel(store1)) {
				Print("the write after rLab(rmw)", store1, "should be also included");
				toBeKept->updateIdx(store1);
			}
			Print("included store1(after rLab)", *toBeKept);
			// getchar();
			// if other rmw reads from store, let it read from e2
			for (auto i = 0u; i < g.getNumThreads(); i++) {
				for (auto j = 0u; j < g.getThreadSize(i); j++) {
					Event e(i, j);
					const EventLabel* lab = g.getEventLabel(e);
					if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab); rrLab && isRMW(rrLab, g)) {
						if (auto* ssLab = rrLab->getRf(); ssLab && ssLab->getPos() == store) {
							auto load2 = rrLab->getPos();
							Print("found rmw (was reading from rLab) to be reordered:", load2);
							// BUG_ON(reorderedRMW);
							reorderedRMW = { load2, store1 };
						}
					}
				}
			}

		}
		// not mutated so far

		Print("getting copy up to:", *toBeKept);
		auto og = g.getCopyUpTo(*toBeKept);
		Print("copied og:", *og);
		// getchar();
		// change rf first, then check

		og->changeRf(rLab->getPos(), store);
		Print("changed rf for rLab", rLab->getPos(), *og);
		// getchar();
		// filling in hole
		if (!isRMW(rLab, g)) {
			Print("rLab", rLab->getPos(), "is not rmw, push and return");
			pushGraph(og);
			return;
		}


		{
			rLab = og->getReadLabel(rLab->getPos());	// label of copied graph
			BUG_ON(!rLab);
			Print("since rLab is rmw, repair the write after it");
			const auto store1 = rLab->getPos().next();
			auto rsLab = og->getWriteLabel(store1);
			BUG_ON(!rsLab);

			og->setEventLabel(store1, og->createHoleLabel(store1));
			og->removeStoreFromCO(rsLab);
			Print("filled in hole", *og);
			// getchar();
		}


		// first complete the write part of the mutated rmw
		auto v = std::make_unique<View>();

		// Event ;	// the write after the mutated read
		if (auto* rsLab = completeRevisitedRMW(og->getReadLabel(rLab->getPos()), *og)) {
			Print("completed the write part of the mutated rmw", *og);
			// getchar();
			pushGraph(og);

		}
		else {
			BUG_ON(!rsLab);
		}

		printPrefixViewsOnStack("before reordering");

		if (reorderedRMW) {
			auto& gg = getGraph();
			Print("after push, g = getGraph():", gg);

			auto [load2, store1] = *reorderedRMW;


			if (llvm::isa<CasReadLabel>(gg.getEventLabel(load2))) {
				// ignoring cas for now
				popExecution();
				return;
			}


			Print("reordering rmw for load2:", load2, ", let it read from store1:", store1);
			// getchar();
			auto* rsLab = gg.getWriteLabel(store1);
			auto* rLab2 = gg.getReadLabel(load2);
			BUG_ON(!rLab2);
			BUG_ON(!rsLab);

			updateLabelViews(rsLab);			// this update is performed on the Graph of getGraph(), didn't work in previous completeRevisitedRMW
			// compute a second time BEFORE change rf for the second time
			auto rsLab2 = gg.getWriteLabel(load2.next());
			// auto rLab = llvm::dyn_cast<ReadLabel>(lab);
			// auto rsLab = llvm::dyn_cast<ReadLabel>(og->getEventLabel(s));
			// rsLab->reset();	// ?????????????????
			// BUG_ON(!rLab);
			BUG_ON(!rsLab2);
			Print("changing rf for load2", load2);
			gg.changeRf(load2, rsLab->getPos());
			repairDanglingReads(gg);
			Print("rf changed, gg = ", gg);

			const auto store2 = load2.next();
			Print(gg, "\nfixing store2", store2);
			BUG_ON(!rsLab2);
			// Print("remove store", rsLab2->getPos(), "from co");
			// g.removeStoreFromCO(rsLab2);
			// Print("adding hole");
			// for (auto j = g.getThreadSize(store2.thread) - 1; j >= store2.index; j--) {
			// 	Event e(store2.thread, j);
			// 	if (auto s = g.getWriteLabel(e)) {
			// 		g.removeStoreFromCO(s);
			// 	}
			// 	Print("holing", e);
			// 	// Print("try including", e);
			// 	const EventLabel* lab = g.getEventLabel(e);
			// 	g.setEventLabel(e, g.createHoleLabel(e));
			// }
			// gg.setEventLabel(store2, gg.createHoleLabel(store2));
			// Print("completing revisit rmw for", rLab2->getPos(), "graph:", gg);
			// completeRevisitedRMW(rLab2);
			// Print("before updating view, load2's view:", getPrefixView(rLab2));
			// updateLabelViews(rLab2);
			// Print("after updating view, load2's view:", getPrefixView(rLab2));
			// getchar();
			// Print("before updating view, store2's view:", getPrefixView(rsLab2));
			// updateLabelViews(rLab2);
			// Print("after updating view, store2's view:", getPrefixView(rsLab2));
			// getchar();
			// repairDanglingReads(*og);



			updateAllViews();



			// Print("after updating all views:", gg);
			// includeNonPoRfSuccessor(rsLab2, gg, v);	
			includeNonPoRfSuccessor(rLab2, gg, v);
			// v->update(getPrefixView(rsLab2));
			v->updateIdx(rsLab->getPos());
			v->updateIdx(rsLab2->getPos());
			Print("cut view of the second time:", *v);
			// getchar();
			// Print("og:", "threads:", og->getNumThreads(), ", ")
			auto og = gg.getCopyUpTo(*v);


			// // 
			Print("cutting for the second time, before reordering", *og);

			// getchar();
			popExecution();
			// Print("poped execution");
			pushGraph(og);
			// updateAllViews();
			{
				Print("fixing store2", store2);
				auto& ggg = getGraph();
				auto rsLab2 = ggg.getWriteLabel(store2);
				auto* rLab2 = ggg.getReadLabel(load2);
				BUG_ON(!rsLab2);
				BUG_ON(!rLab2);
				std::vector<Event> readers;
				for (auto i = 0u; i < g.getNumThreads(); i++) {
					for (auto j = 0u; j < g.getThreadSize(i); j++) {
						Event e(i, j);
						const EventLabel* lab = g.getEventLabel(e);
						if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab); rrLab && rrLab->getRf()) {
							if (rrLab->getRf()->getPos() == store2) {
								readers.push_back(e);
							}
						}
					}
				}
				// manually recompute the rmw result:

				ggg.setEventLabel(store2, ggg.createHoleLabel(store2));
				ggg.removeStoreFromCO(rsLab2);
				Print("after adding hole:", ggg);
				// rLab2->result
				// rsLab2->result
				if (!completeRevisitedRMW(rLab2, ggg)
					// || readers.size()

					) {
					Print("cas value != expected");
					popExecution();
					return;
				}

				repairDanglingReads(ggg);
				Print("before update all views: ggg", ggg);
				updateAllViews();
				Print("readded the store2, ggg", ggg);

			}
		}
	}
}

// maximal + reordering rmw
template<bool PushOnce>
void GenMCDriver::maximalCutRMW2() {

	const auto& g = getGraph();
	// pick loads with multiple store options
	using RFPair = std::pair<const ReadLabel* const, Event>;
	std::vector<RFPair> rf_options;



	auto pick = [](const EventLabel* lab, const ExecutionGraph& g) -> const ReadLabel*
		{
			auto rLab = llvm::dyn_cast<ReadLabel>(lab);
			return rLab && rLab->getRf()
				&& !rLab->isNotAtomic() ? rLab : nullptr;
		};




	auto rmv = [&](const ReadLabel* rLab, const ExecutionGraph& gg)
		{

			return [&gg, rLab, this](const Event& e) -> bool
				{
					auto lstamp = rLab->getStamp();
					auto sLab = gg.getWriteLabel(e);
					auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };
					// Print("prefix of", e, ":", getPrefixView(g.getEventLabel(e)));

					return e == rLab->getRf()->getPos()	// remove the current rf
						|| getPrefixView(gg.getEventLabel(e)).contains(rLab)// e.thread == rLab->getThread()

						;
					;
				};
		};
	if constexpr (PushOnce) {
		if (auto rf = selectAlternativeRF(g, pick, rmv)) {
			do_maximalCutRMW2(*rf, g);
		}
	}
	else {
		// do_maximalCutRMW2 also changes the internal state, can't push multiple times...
		// if (auto rfs = selectAlternativeRF_n(g, pick, rmv, push_num); rfs.size()) {
		// 	for (auto&& rf : rfs) do_maximalCutRMW2(rf, g);
		// }
		if (auto rf = selectAlternativeRF(g, pick, rmv)) {
			do_maximalCutRMW2(*rf, g);
		}
	}

}







void GenMCDriver::mixCutDemo() {
	static EventCounter cnt(getenv("TEST"));
	cnt.n_graph++;
	// getchar();
	const auto& g = getGraph();
	// pick loads with multiple store options
	using RFPair = std::pair<const ReadLabel* const, Event>;
	std::vector<RFPair> rf_options;


	auto pick = [](const EventLabel* lab, const ExecutionGraph& g) -> const ReadLabel*
		{
			auto rLab = llvm::dyn_cast<ReadLabel>(lab);
			return rLab && rLab->getRf()
				&& !rLab->isNotAtomic() ? rLab : nullptr;
		};




	auto rmv = [&](const ReadLabel* rLab, const ExecutionGraph& gg)
		{

			return [&gg, rLab, this](const Event& e) -> bool
				{
					auto lstamp = rLab->getStamp();
					auto sLab = gg.getWriteLabel(e);
					auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };
					// Print("prefix of", e, ":", getPrefixView(g.getEventLabel(e)));

					return e == rLab->getRf()->getPos()	// remove the current rf
						|| getPrefixView(gg.getEventLabel(e)).contains(rLab)// e.thread == rLab->getThread()

						;
					;
				};
		};




	if (auto rf = selectAlternativeRF(g, pick, rmv)) {

		auto rf_to_chg = *rf;
		auto* rLab = rf_to_chg.first;
		BUG_ON(!rLab);
		auto store = rf_to_chg.second;
		Print("rf selected: ", *rLab, "option: ", store);



		// getchar();

		// get stamps and do forward/backward revist
		auto lstamp = rLab->getStamp();
		auto sLab = g.getWriteLabel(store);

		BUG_ON((store != Event{ 0,0 } ? !sLab : false));
		// 
		// if (!sLab) return;
		auto sstamp = sLab ? sLab->getStamp() : Stamp{ 0 };
		Print("load stamp: ", lstamp, ", store stamp: ", sstamp);

		if (sLab) {
			// total events num:
			size_t total = 0;
			for (auto i = 0u; i < g.getNumThreads(); i++) {
				total += g.getThreadSize(i);
			}
			for (int i = 0; i < 3; i++) {
				cnt.n_totalEvents[i] += total;
			}

			constexpr int REVISIT = 0;
			constexpr int MINIMAL = 1;
			constexpr int MAXIMAL = 2;
			// revisit ==================
			{
				if (lstamp < sstamp) {
					auto br = constructBackwardRevisit(rLab, sLab);
					auto v = br->getViewRel();
					for (int i = 0; i < v->size(); i++) {
						cnt.n_remainedEvents[REVISIT] += v->getMax(i);
					}

				}
				else {
					cnt.n_remainedEvents[REVISIT] += lstamp.get();
				}
			}

			// minimal -------------------
			{
				auto porf_s = getPrefixView(sLab).clone();
				Print("porf_s:", *porf_s);
				auto porf_r = getPrefixView(rLab).clone();
				Print("porf_r:", *porf_r);
				// porf_r.update(porf_s);
				auto& v = porf_s->update(*porf_r);
				for (int i = 0; i < v.size(); i++) {
					cnt.n_remainedEvents[MINIMAL] += v.getMax(i);
				}
			}

			// maximal ~~~~~~~~~~~~~~~~~~~~
			{
				auto toBeKept = std::make_unique<View>();

				for (auto i = 0u; i < g.getNumThreads(); i++) {
					for (auto j = 0u; j < g.getThreadSize(i); j++) {
						Event e(i, j);

						const EventLabel* lab = g.getEventLabel(e);
						if (!getPrefixView(lab).contains(rLab)) {
							bool updateFlag = true;
							if (auto jLab = llvm::dyn_cast<ThreadJoinLabel>(lab)) {
								if (auto* lLab = llvm::dyn_cast<ThreadFinishLabel>(g.getLastThreadLabel(jLab->getChildId()))) {
									if (getPrefixView(lLab).contains(rLab)) {
										updateFlag = false;
									}
									else {
										// toBeKept->updateIdx(jLab->getPos());
									}
								}
							}
							if (auto rrLab = llvm::dyn_cast<ReadLabel>(lab)) {
								if (auto* ssLab = rrLab->getRf()) {
									if (getPrefixView(ssLab).contains(rLab)) {
										updateFlag = false;
									}

								}
							}

							if (updateFlag) {
								toBeKept->updateIdx(e);
							}


						}
					}
				}
				for (int i = 0; i < toBeKept->size(); i++) {
					cnt.n_remainedEvents[MAXIMAL] += toBeKept->getMax(i);
				}
			}

		}
	}
}

void GenMCDriver::mutateAndPush() {

	switch (mut)
	{
	case 0:
		// do nothing
		break;
	case 1:
		revisitCut();
		break;
	case 2:
		minimalCut();
		break;
	case 25:
	case 21:	// push several times, but do not count freq
		minimalCut_n();	// minimal cut for several times
		break;
	case 3:
		maximalCut();
		break;
	case 35:	// maximalcut with frequency 
		maximalCut_n();	// not used
		break;
	case 4:
		maximalCutRMW();
		break;
	case 5:
		mixCutDemo();
		break;
	case 6:
		maximalCut2();
		break;
	case 65:
		maximalCut2<false>();
		break;
	case 7:
		powerSched();
		break;
	case 8:
		maximalCutRMW2();
		break;
	case 85:
		maximalCutRMW2<false>();	// maximal cut + reorder rmw 
		break;
		// various cuts with powerscheds:
	case 27:
		minimalCut_n(true);
		break;
	case 37:
		maximalCutRMW2<false>();
		break;
	default:
		Print("mut = ", mut);
		BUG();
	}

}

#endif

// GraphCounter gcounter{};

void GenMCDriver::handleExecutionEnd()
{
#ifdef DEBUG_LUAN
	Print(RED("handleEnd: graph:\n"), getGraph());
	// Print("handleEnd:");
	// printGraph(false, llvm::outs());
	Print("is blocked:", isExecutionBlocked() ? "true" : "false");
	// getchar();
	// Print("handleEnd print finished\n");
	// auto& g = getGraph();
	static bool init_flag = false;
	if (!init_flag) {
		init_flag = true;
		static const auto test = getenv("TEST");
		BUG_ON(!test);
		static GraphCounter gcounter(getConf()->estimationMax, test);
		this->gc = &gcounter;
	}
	auto& gcounter = getGraphCounter();
#endif
#ifdef DEBUG_LUAN
	bool not_interesting = gcounter.contains(getGraph());
	auto h = gcounter.log(getGraph(), isExecutionBlocked());
	// gcounter.print_gmap();
	if (gcounter.plot.size() > getConf()->estimationMax) {
		// shouldHalt = true;
	}
#endif

	if (mut == 7 || mut == 27 || mut == 37) {
		gcounter.updateWeights(!not_interesting, h);
	}

	static float miu = 0.005;
	// {
	// 	const auto N = execStack.size();
	// 	auto f = (execStack.size() / float(N) - 0.1);
	// 	if (f > 0) miu -= std::max<float>(0.1, std::abs(f));
	// 	else miu += std::max<float>(0.1, std::abs(f));
	// 	if (miu < 0) miu = 0.01;
	// }

	if (
		// mut == 7
		// || mut == 25
		// || mut == 85
		// || mut == 35
		// || mut == 65
		// || mut == 17
		// || mut == 27
		// || mut == 37
		true
		) {
		if (not_interesting && gcounter.relative_freq(h) < miu) not_interesting = false;
	}

	// limit the execStack size:
	if (
		mut == 25
		// || mut == 85
		|| mut == 65
		) {
		if (execStack.size() > 10) {
			// pop old executions
			for (int i = 0; i < 4; i++) {
				execStack.pop_front();
				// not_interesting = true;
			}
		}
	}


#ifdef FUZZ_LUAN

	if (not_interesting) {
		Print(BLUE("contains g"), gcounter.ghash(getGraph()));
		// getchar();
	}
	else {
		mutateAndPush();
	}
#endif



	/* LAPOR: Check lock-well-formedness */
	if (getConf()->LAPOR && !isLockWellFormedLAPOR())
		WARN_ONCE("lapor-not-well-formed", "Execution not lock-well-formed!\n");

	if (isMoot()) {
		GENMC_DEBUG(++result.exploredMoot; );
		return;
	}

	/* Helper: Check helping CAS annotation */
	if (getConf()->helper)
		checkHelpingCasAnnotation();

	/* If under estimation mode, guess the total.
	 * (This may run a few times, but that's OK.)*/

	if (inEstimationMode()) {
		updateStSpaceEstimation();
		if (!shouldStopEstimating()) {
			addToWorklist(0, std::make_unique<RerunForwardRevisit>());
		}
	}

	/* Ignore the execution if some assume has failed */
	if (isExecutionBlocked()) {
		++result.exploredBlocked;
		if (getConf()->printBlockedExecs)
			printGraph();
		if (getConf()->checkLiveness)
			checkLiveness();
		return;
	}

	if (getConf()->printExecGraphs && !getConf()->persevere)
		printGraph(); /* Delay printing if persevere is enabled */
	++result.explored;
}

void GenMCDriver::handleRecoveryStart()
{
	if (isExecutionBlocked())
		return;

	auto& g = getGraph();
	auto* EE = getEE();

	/* Make sure that a thread for the recovery routine is
	 * added only once in the execution graph*/
	if (g.getRecoveryRoutineId() == -1)
		g.addRecoveryThread();

	/* We will create a start label for the recovery thread.
	 * We synchronize with a persistency barrier, if one exists,
	 * otherwise, we synchronize with nothing */
	auto tid = g.getRecoveryRoutineId();
	auto psb = g.collectAllEvents([&](const EventLabel* lab)
		{ return llvm::isa<DskPbarrierLabel>(lab); });
	if (psb.empty())
		psb.push_back(Event::getInit());
	ERROR_ON(psb.size() > 1, "Usage of only one persistency barrier is allowed!\n");

	auto tsLab = ThreadStartLabel::create(Event(tid, 0), psb.back(), ThreadInfo(tid, psb.back().thread, 0, 0));
	auto* lab = addLabelToGraph(std::move(tsLab));

	/* Create a thread for the interpreter, and appropriately
	 * add it to the thread list (pthread_create() style) */
	EE->createAddRecoveryThread(tid);

	/* Finally, do all necessary preparations in the interpreter */
	getEE()->setupRecoveryRoutine(tid);
	return;
}

void GenMCDriver::handleRecoveryEnd()
{
	/* Print the graph with the recovery routine */
	if (getConf()->printExecGraphs)
		printGraph();
	getEE()->cleanupRecoveryRoutine(getGraph().getRecoveryRoutineId());
	return;
}

void GenMCDriver::run()
{
	/* Explore all graphs and print the results */
	explore();
}

bool GenMCDriver::isHalting() const
{

	auto* tp = getThreadPool();
	return shouldHalt || (tp && tp->shouldHalt());
}

void GenMCDriver::halt(VerificationError status)
{
#ifdef DEBUG_LUAN
	Print(RED("halt()"), status);
	Print(getGraph());
	if (gc) {
		gc->bug();
	}
	// getchar();
#endif
	getEE()->block(BlockageType::Error);

	shouldHalt = true;
#ifdef DEBUG_LUAN
	shouldHalt = false;
#endif
	result.status = status;
	if (getThreadPool())
		getThreadPool()->halt();


}

GenMCDriver::Result GenMCDriver::verify(std::shared_ptr<const Config> conf, std::unique_ptr<llvm::Module> mod, std::unique_ptr<ModuleInfo> modInfo)
{
	/* Spawn a single or multiple drivers depending on the configuration */
	if (conf->threads == 1) {
		auto driver = DriverFactory::create(conf, std::move(mod), std::move(modInfo));
		driver->run();
		return driver->getResult();
	}

	std::vector<std::future<GenMCDriver::Result>> futures;
	{
		/* Then, fire up the drivers */
		ThreadPool pool(conf, mod, modInfo);
		futures = pool.waitForTasks();
	}

	GenMCDriver::Result res;
	for (auto& f : futures) {
		res += f.get();
	}
	return res;
}

GenMCDriver::Result GenMCDriver::estimate(std::shared_ptr<const Config> conf,
	const std::unique_ptr<llvm::Module>& mod,
	const std::unique_ptr<ModuleInfo>& modInfo)
{
	auto estCtx = std::make_unique<llvm::LLVMContext>();
	auto newmod = LLVMModule::cloneModule(mod, estCtx);
	auto newMI = modInfo->clone(*newmod);
	auto driver = DriverFactory::create(conf, std::move(newmod), std::move(newMI), GenMCDriver::EstimationMode{ conf->estimationMax });
	driver->run();
#ifdef DEBUG_LUAN
	auto res = driver->getResult();
	driver.release();
	return res;
#endif
	return driver->getResult();
}

void GenMCDriver::addToWorklist(Stamp stamp, WorkSet::ItemT item)
{
#ifdef FUZZ_LUAN
	if (stamp == 0) {
		if (auto* rr = llvm::dyn_cast<RerunForwardRevisit>(&*item)) {

			Print("adding rerun");
			if (getWorkqueue()[stamp].size() >= 2) {
				Print(GREEN("work queue: "), "not empty");
				for (auto&& p : getWorkqueue()) {
					Print(p.first, ", ", p.second);
				}
				// getchar();
				return;
			}
		}
	}
#endif
	getWorkqueue()[stamp].add(std::move(item));
}

std::pair<Stamp, WorkSet::ItemT>
GenMCDriver::getNextItem()
{

	auto& workqueue = getWorkqueue();
#ifdef DEBUG_LUAN
	Print("GenMCDriver::getNextItem() " "work queue: ");
	for (auto&& p : workqueue) {
		Print(p.first, ", ", p.second);
	}
	// getchar();
#endif
	for (auto rit = workqueue.rbegin(); rit != workqueue.rend(); ++rit) {
		if (rit->second.empty()) {
			continue;
		}

		return { rit->first, rit->second.getNext() };
	}
	return { 0, nullptr };
}


/************************************************************
 ** Scheduling methods
 ***********************************************************/

void GenMCDriver::blockThread(Event pos, BlockageType t)
{
	/* There are a couple of reasons we don't call Driver::addLabelToGraph() here:
	 *   1) It's redundant to update the views of the block label
	 *   2) If addLabelToGraph() does extra stuff (e.g., event caching) we absolutely
	 *      don't want to do that here. blockThread() should be safe to call from
	 *      anywhere in the code, with no unexpected side-effects */
	getGraph().addLabelToGraph(BlockLabel::create(pos, t));
	getEE()->getThrById(pos.thread).block(t);
}

void GenMCDriver::blockThreadTryMoot(Event pos, BlockageType t)
{
	blockThread(pos, t);
	mootExecutionIfFullyBlocked(pos);
}

void GenMCDriver::unblockThread(Event pos)
{
	auto* bLab = getGraph().getLastThreadLabel(pos.thread);
	BUG_ON(!llvm::isa<BlockLabel>(bLab));
	getGraph().removeLast(pos.thread);
	getEE()->getThrById(pos.thread).unblock();
}

bool GenMCDriver::scheduleAtomicity()
{
	auto* lastLab = getGraph().getEventLabel(lastAdded);
	if (llvm::isa<FaiReadLabel>(lastLab)) {
		getEE()->scheduleThread(lastAdded.thread);
		return true;
	}
	if (auto* casLab = llvm::dyn_cast<CasReadLabel>(lastLab)) {
		if (getReadValue(casLab) == casLab->getExpected()) {
			getEE()->scheduleThread(lastAdded.thread);
			return true;
		}
	}
	return false;
}

bool GenMCDriver::scheduleNormal()
{
	if (inEstimationMode())
		return scheduleNextWFR();	// pick next thread rand

	switch (getConf()->schedulePolicy) {
	case SchedulePolicy::ltr:
		return scheduleNextLTR();
	case SchedulePolicy::wf:
		return scheduleNextWF();
	case SchedulePolicy::wfr:
		return scheduleNextWFR();
	case SchedulePolicy::arbitrary:
		return scheduleNextRandom();
	default:
		BUG();
	}
	BUG();
}

bool GenMCDriver::rescheduleReads()
{
	auto& g = getGraph();
	auto* EE = getEE();

	for (auto i = 0u; i < g.getNumThreads(); ++i) {
		auto* bLab = llvm::dyn_cast<BlockLabel>(g.getLastThreadLabel(i));
		if (!bLab || bLab->getType() != BlockageType::ReadOptBlock)
			continue;

		setRescheduledRead(bLab->getPos());
		unblockThread(bLab->getPos());
		EE->scheduleThread(i);
		return true;
	}
	return false;
}



bool GenMCDriver::scheduleNext()
{
#ifdef DEBUG_LUAN
	// Print("GenMCDriver::scheduleNext()");
	// Print("isMoot? :", isMoot() ? "true" : "false");
	// Print("isHalting? :", isHalting() ? "true" : "false");
	// Print("schedule policy: ", int(getConf()->schedulePolicy));
	// Print(g);
	// getchar();
#endif
	if (isMoot() || isHalting())
		return false;

	auto& g = getGraph();
	auto* EE = getEE();


	/* 1. Ensure atomicity. This needs to here because of weird interactions with in-place
	 * revisiting and thread priotitization. For example, consider the following scenario:
	 *     - restore @ T2, in-place rev @ T1, prioritize rev @ T1,
	 *       restore FAIR @ T2, schedule T1, atomicity violation */
	if (scheduleAtomicity())
		return true;

	/* Check if we should prioritize some thread */
	if (schedulePrioritized())
		return true;

	/* Schedule the next thread according to the chosen policy */
	if (scheduleNormal())
		return true;

	/* Finally, check if any reads needs to be rescheduled */
	return rescheduleReads();
}

std::vector<ThreadInfo> createExecutionContext(const ExecutionGraph& g)
{
	std::vector<ThreadInfo> tis;
	for (auto i = 1u; i < g.getNumThreads(); i++) { // skip main
		auto* bLab = g.getFirstThreadLabel(i);
		BUG_ON(!bLab);
		tis.push_back(bLab->getThreadInfo());
	}
	return tis;
}


#ifdef DEBUG_LUAN


void GenMCDriver::explore_dbg() {


	// auto* EE = getEE();
	// EE->setExecutionContext(createExecutionContext(gg)); 
	// EE->reset();


	// constexpr int N = 1e4;
	// const auto& inputf = getConf()->inputFile;
	// GraphCounter<N> counter(inputf);

	// for (auto i = 0; i < N; i++) {
	// 	EE->reset();


	// 	resetExplorationOptions();

	/* Get main program function and run the program */
	// Print("before runasMain:\n", gg);
	// Print("prog entry: ", getConf()->programEntryFun);
	// this->lastAdded = Event::getInit();
	// Print("lastAdded = ", lastAdded);
	// Print("should halt = ", shouldHalt ? "true" : "false");
	// getchar();

	// EE->runAsMain(getConf()->programEntryFun);

	// Print("after runasMain:\n", gg);
	// getchar();

	// if (getConf()->persevere)
	// 	EE->runRecovery();

	// graph_freq[ghash(gg)]++;
	// print_gmap(graph_freq);
	// counter.log(gg);
	// getExecution().restrict(Stamp{ 0 });
	// Print("after restricting graph:\n", gg);
	// getchar();

	// auto rerun = std::make_unique<RerunForwardRevisit>();

	// }
	// auto& gg = getGraph();

	// local clock for testing stop-on-timeout
	auto myBegin = std::chrono::high_resolution_clock::now();
	auto getElapsedSecs(const std::chrono::high_resolution_clock::time_point & begin) -> long double;


	auto* EE = getEE();

	resetExplorationOptions();
	EE->setExecutionContext(createExecutionContext(getGraph()));
	while (
#ifdef CATCHSIG_LUAN
		// Print("check if isHalting, shouldHalt =", shouldHalt ? "true" : "false"),
#endif
		// !isHalting()
		result.explored + result.exploredBlocked < getConf()->estimationMin
		) {
		if (getElapsedSecs(myBegin) > time_out) {
			return;
		}
		if (gc) {
			if (gc->plot.size() >= getConf()->estimationMin) {
				return;
			}
			else {
				// check which prefix the current graph is
				if (mut == 7
					|| mut == 27 || mut == 37
					) gc->locatePrefix(getGraph());
			}

			// #ifdef HALT_ON_ERROR
			if (halt_on_error)
				if (gc->bug_iters.size()) return;
			// #endif
		}

		EE->reset();
		// Print("!is halting, rerun as main. graph: \n", getGraph());

#ifdef DEBUG_LUAN
		// const auto& workqueue = getWorkqueue();
		Print("before rerun as main work queue: ");
		for (auto&& p : getWorkqueue()) {
			Print(p.first, ", ", p.second);
		}
		// Print(RED("execStack.size() = "), execStack.size());
		// getchar();
		// Print(GREEN("graph"), getGraph());

#endif
		/* Get main program function and run the program */
		BUG_ON(!getConf());
		EE->runAsMain(getConf()->programEntryFun);


		// Print("after run as main\n");
		// getchar();
		if (getConf()->persevere)
			EE->runRecovery();

		auto validExecution = false;
		while (!validExecution
			// || !getWorkqueue().empty()
			) {
			/*
			 * restrictAndRevisit() might deem some execution infeasible,
			 * so we have to reset all exploration options before
			 * calling it again
			 */
			resetExplorationOptions();

			auto [stamp, item] = getNextItem();
			if (!item) {
#ifdef FUZZ_LUAN
				Print("!item..");
#endif
				if (popExecution()) {
					Print("popExecution() = true");
					continue;
				}

				return;
			}
			BUG_ON(!item);
			auto pos = item->getPos();
			Print("revisiting item: ", pos);
			// getchar();
			validExecution = restrictAndRevisit(stamp, item) && isRevisitValid(*item);
			// Print("after revisit: ", gg);
			Print("validExecution = ", validExecution ? YELLOW("true") : GREEN("false"));
			if (validExecution) {

				// popExecution();
			}

			Print("work queue: ");
			for (auto&& p : getWorkqueue()) {
				Print(p.first, ", ", p.second);
			}
			// getchar();
		}

	}
}

#endif

void GenMCDriver::explore()
{
#ifdef DEBUG_LUAN
	explore_dbg();
	return;
#endif

	auto* EE = getEE();

	resetExplorationOptions();
	EE->setExecutionContext(createExecutionContext(getGraph()));
	while (!isHalting()) {
		EE->reset();

		/* Get main program function and run the program */
		EE->runAsMain(getConf()->programEntryFun);
		if (getConf()->persevere)
			EE->runRecovery();

		auto validExecution = false;
		while (!validExecution) {
			/*
			 * restrictAndRevisit() might deem some execution infeasible,
			 * so we have to reset all exploration options before
			 * calling it again
			 */
			resetExplorationOptions();

			auto [stamp, item] = getNextItem();
			if (!item) {
				if (popExecution())
					continue;
				return;
			}
			auto pos = item->getPos();
			validExecution = restrictAndRevisit(stamp, item) && isRevisitValid(*item);
		}
	}
}

bool readsUninitializedMem(const ReadLabel* lab)
{
	return lab->getAddr().isDynamic() && lab->getRf()->getPos().isInitializer();
}

bool GenMCDriver::isRevisitValid(const Revisit& revisit)
{
	auto& g = getGraph();
	auto pos = revisit.getPos();
	auto* mLab = llvm::dyn_cast<MemAccessLabel>(g.getEventLabel(pos));

	/* E.g., for optional revisits, do nothing */
	if (!mLab)
		return true;

	if (!isExecutionValid(mLab))
		return false;

	auto* rLab = llvm::dyn_cast<ReadLabel>(mLab);
	if (rLab && readsUninitializedMem(rLab)) {
		reportError(pos, VerificationError::VE_UninitializedMem);
		return false;
	}

	/* If an extra event is added, re-check consistency */
	return g.isRMWLoad(pos) ? isExecutionValid(g.getNextLabel(pos)) : true;
}

bool GenMCDriver::isExecutionDrivenByGraph(const EventLabel* lab)
{
	const auto& g = getGraph();
	auto curr = lab->getPos();
	auto replay = (curr.index < g.getThreadSize(curr.thread)) &&
		!llvm::isa<EmptyLabel>(g.getEventLabel(curr));
	if (!replay && !llvm::isa<MallocLabel>(lab) && !llvm::isa<ReadLabel>(lab))
		cacheEventLabel(lab);
	return replay;
}

bool GenMCDriver::inRecoveryMode() const
{
	return getEE()->getProgramState() == llvm::ProgramState::Recovery;
}

bool GenMCDriver::inReplay() const
{
	return getEE()->getExecState() == llvm::ExecutionState::Replay;
}

EventLabel* GenMCDriver::addLabelToGraph(std::unique_ptr<EventLabel> lab)
{
	auto& g = getGraph();
	auto* addedLab = g.addLabelToGraph(std::move(lab));
	updateLabelViews(addedLab);
	lastAdded = addedLab->getPos();
	if (addedLab->getIndex() >= getConf()->warnOnGraphSize) {
		LOG_ONCE("large-graph", VerbosityLevel::Tip)
			<< "The execution graph seems quite large. "
			<< "Consider bounding all loops or using -unroll\n";
	}
	return addedLab;
}

#ifdef REORDER_RMW
EventLabel* GenMCDriver::addLabelToGraph(std::unique_ptr<EventLabel> lab, ExecutionGraph& g)
{
	auto* addedLab = g.addLabelToGraph(std::move(lab));
	updateLabelViews(addedLab);
	lastAdded = addedLab->getPos();
	if (addedLab->getIndex() >= getConf()->warnOnGraphSize) {
		LOG_ONCE("large-graph", VerbosityLevel::Tip)
			<< "The execution graph seems quite large. "
			<< "Consider bounding all loops or using -unroll\n";
	}
	return addedLab;
}
#endif

void GenMCDriver::updateLabelViews(EventLabel* lab)
{
	updateMMViews(lab);
	if (!getConf()->symmetryReduction)
		return;

	auto& v = lab->getPrefixView();
	updatePrefixWithSymmetriesSR(lab);
}

VerificationError GenMCDriver::checkForRaces(const EventLabel* lab)
{
	if (getConf()->disableRaceDetection || inEstimationMode())
		return VerificationError::VE_OK;

	/* Check for hard errors */
	const EventLabel* racyLab = nullptr;
	auto err = checkErrors(lab, racyLab);
	if (err != VerificationError::VE_OK) {
		reportError(lab->getPos(), err, "", racyLab);
		return err;
	}

	/* Check whether there are any unreported warnings... */
	std::vector<const EventLabel*> races;
	auto newWarnings = checkWarnings(lab, getResult().warnings, races);
	getResult().warnings.insert(newWarnings.begin(), newWarnings.end());

	/* ... and report them */
	auto i = 0U;
	for (auto& wcode : newWarnings) {
		auto hardError = (getConf()->symmetryReduction || getConf()->ipr) && wcode == VerificationError::VE_WWRace;
		auto msg = hardError ? "Warning treated as an error due to symmetry reduction/in-place revisiting.\n"
			"You can use -disable-sr and -disable-ipr to disable these features."s : ""s;
		reportError(lab->getPos(), wcode, msg, races[i++], hardError);
		if (hardError)
			return wcode;
	}
	return VerificationError::VE_OK;
}

void GenMCDriver::cacheEventLabel(const EventLabel* lab)
{
	if (!getConf()->instructionCaching || inEstimationMode())
		return;

	auto& g = getGraph();

	/* Extract value prefix and cached data */
	auto [vals, last] = extractValPrefix(lab->getPos());
	auto* data = retrieveCachedSuccessors(lab->getThread(), vals);

	/*
	 * Check if there are any new data to cache.
	 * (For dep-tracking, we could optimize toIdx and collect until
	 * a new (non-empty) label with a value is found.)
	 */
	auto fromIdx = (!data || data->empty()) ? last.index : data->back()->getIndex();
	auto toIdx = lab->getIndex();
	if (data && !data->empty() && data->back()->getIndex() >= toIdx)
		return;

	/*
	 * Go ahead and collect the new data. We have to be careful when
	 * cloning LAB because it has not been added to the graph yet.
	 */
	std::vector<std::unique_ptr<EventLabel>> labs;
	for (auto i = fromIdx + 1; i <= toIdx; i++) {
		auto cLab = (i == lab->getIndex()) ? lab->clone() : g.getEventLabel(Event(lab->getThread(), i))->clone();
		cLab->reset();
		labs.push_back(std::move(cLab));
	}

	/* Is there an existing entry? */
	if (!data) {
		auto res = seenPrefixes[lab->getThread()].addSeq(vals, std::move(labs));
		BUG_ON(!res);
		return;
	}

	BUG_ON(data->empty() && last.index >= lab->getIndex());
	BUG_ON(!data->empty() && data->back()->getIndex() + 1 != lab->getIndex());

	data->reserve(data->size() + labs.size());
	std::move(std::begin(labs), std::end(labs), std::back_inserter(*data));
	labs.clear();
}

/* Given an event in the graph, returns the value of it */
SVal GenMCDriver::getWriteValue(const EventLabel* lab, const AAccess& access)
{
	/* If the even represents an invalid access, return some value */
	if (!lab)
		return SVal();

	/* If the event is the initializer, ask the interpreter about
	 * the initial value of that memory location */
	if (lab->getPos().isInitializer())
		return getEE()->getLocInitVal(access);

	/* Otherwise, we will get the value from the execution graph */
	auto* wLab = llvm::dyn_cast<WriteLabel>(lab);
	BUG_ON(!wLab);

	/* It can be the case that the load's type is different than
	 * the one the write's (see troep.c).  In any case though, the
	 * sizes should match */
	if (wLab->getSize() != access.getSize())
		reportError(wLab->getPos(), VerificationError::VE_MixedSize,
			"Mixed-size accesses detected: tried to read event with a " +
			std::to_string(access.getSize().get() * 8) + "-bit access!\n" +
			"Please check the LLVM-IR.\n");

	/* If the size of the R and the W are the same, we are done */
	return wLab->getVal();
}

/* Same as above, but the data of a file are not explicitly initialized
 * so as not to pollute the graph with events, since a file can be large.
 * Thus, we treat the case where WRITE reads INIT specially. */
SVal GenMCDriver::getDskWriteValue(const EventLabel* lab, const AAccess& access)
{
	if (lab->getPos().isInitializer())
		return SVal();
	return getWriteValue(lab, access);
}

SVal GenMCDriver::getJoinValue(const ThreadJoinLabel* jLab) const
{
	auto& g = getGraph();
	auto* lLab = llvm::dyn_cast<ThreadFinishLabel>(g.getLastThreadLabel(jLab->getChildId()));
	BUG_ON(!lLab);
	return lLab->getRetVal();
}

SVal GenMCDriver::getStartValue(const ThreadStartLabel* bLab) const
{
	auto& g = getGraph();
	if (bLab->getPos().isInitializer() || bLab->getThread() == g.getRecoveryRoutineId())
		return SVal();

	return bLab->getThreadInfo().arg;
}

SVal GenMCDriver::getBarrierInitValue(const AAccess& access)
{
	const auto& g = getGraph();
	auto sIt = std::find_if(store_begin(g, access.getAddr()), store_end(g, access.getAddr()),
		[&access, &g](auto& bLab) {
			BUG_ON(!llvm::isa<WriteLabel>(bLab));
			return bLab.getAddr() == access.getAddr() && bLab.isNotAtomic();
		});

	/* All errors pertinent to initialization should be captured elsewhere */
	BUG_ON(sIt == store_end(g, access.getAddr()));
	return getWriteValue(&*sIt, access);
}

SVal GenMCDriver::getReadRetValueAndMaybeBlock(const ReadLabel* rLab)
{
#ifdef DEBUG_LUAN
	Print("getReadRetValueAndMaybeBlock() for ", rLab->getPos());

#endif
	auto& thr = getEE()->getCurThr();


	/* Fetch appropriate return value and check whether we should block */
	auto res = getReadValue(rLab);

	if (!rLab->getRf()) {
		/* Bottom is an acceptable re-option only @ replay; block anyway */
#ifdef DEBUG_LUAN
#else
		BUG_ON(!inReplay());
#endif
		thr.block(BlockageType::Error);
		Print("block on error");
	}
	else if (llvm::isa<BWaitReadLabel>(rLab) &&
		res != getBarrierInitValue(rLab->getAccess())) {
		/* Reading a non-init barrier value means that the thread should block */
		thr.block(BlockageType::Barrier);
		Print("block on barrier");
	}
	Print("read value:", res);
	return res;
}

SVal GenMCDriver::getRecReadRetValue(const ReadLabel* rLab)
{
	auto& g = getGraph();
	auto rf = g.getLastThreadStoreAtLoc(rLab->getPos(), rLab->getAddr());
	BUG_ON(rf.isInitializer());
	return getWriteValue(g.getEventLabel(rf), rLab->getAccess());
}

bool GenMCDriver::isCoMaximal(SAddr addr, Event e, bool checkCache /* = false */)
{
	return getGraph().isCoMaximal(addr, e, checkCache);
}

bool GenMCDriver::isHazptrProtected(const MemAccessLabel* mLab) const
{
	auto& g = getGraph();
	BUG_ON(!mLab->getAddr().isDynamic());

	auto* aLab = mLab->getAlloc();
	BUG_ON(!aLab);
	auto* pLab = llvm::dyn_cast_or_null<HpProtectLabel>(
		g.getPreviousLabelST(mLab, [&](const EventLabel* lab) {
			auto* pLab = llvm::dyn_cast<HpProtectLabel>(lab);
			return pLab && aLab->contains(pLab->getProtectedAddr());
			}));
	if (!pLab)
		return false;

	for (auto j = pLab->getIndex() + 1; j < mLab->getIndex(); j++) {
		auto* lab = g.getEventLabel(Event(mLab->getThread(), j));
		if (auto* oLab = dyn_cast<HpProtectLabel>(lab))
			if (oLab->getHpAddr() == pLab->getHpAddr())
				return false;
	}
	return true;
}

MallocLabel* findAllocatingLabel(const ExecutionGraph& g, const SAddr& addr)
{
	auto labIt = std::find_if(label_begin(g), label_end(g), [&](auto& lab) {
		auto* mLab = llvm::dyn_cast<MallocLabel>(&lab);
		return mLab && mLab->contains(addr);
		});
	if (labIt != label_end(g))
		return llvm::dyn_cast<MallocLabel>(&*labIt);
	return nullptr;
}

bool GenMCDriver::isAccessValid(const MemAccessLabel* lab) const
{
	/* Make sure that the interperter is aware of this static variable */
	if (!lab->getAddr().isDynamic())
		return getEE()->isStaticallyAllocated(lab->getAddr());

	/* Dynamic accesses are valid if they access allocated memory */
	auto& g = getGraph();
	return !lab->getAddr().isNull() && findAllocatingLabel(g, lab->getAddr());
}

void GenMCDriver::checkLockValidity(const ReadLabel* rLab, const std::vector<Event>& rfs)
{
	auto* lLab = llvm::dyn_cast<LockCasReadLabel>(rLab);
	if (!lLab)
		return;

	/* Should not read from destroyed mutex */
	auto rfIt = std::find_if(rfs.cbegin(), rfs.cend(), [this, lLab](const Event& rf) {
		auto rfVal = getWriteValue(getGraph().getEventLabel(rf), lLab->getAccess());
		return rfVal == SVal(-1);
		});
	if (rfIt != rfs.cend())
		reportError(rLab->getPos(), VerificationError::VE_UninitializedMem,
			"Called lock() on destroyed mutex!", getGraph().getEventLabel(*rfIt));
}

void GenMCDriver::checkUnlockValidity(const WriteLabel* wLab)
{
	auto* uLab = llvm::dyn_cast<UnlockWriteLabel>(wLab);
	if (!uLab)
		return;

	/* Unlocks should unlock mutexes locked by the same thread */
	if (getGraph().getMatchingLock(uLab->getPos()).isInitializer()) {
		reportError(uLab->getPos(), VerificationError::VE_InvalidUnlock,
			"Called unlock() on mutex not locked by the same thread!");
	}
}

void GenMCDriver::checkBInitValidity(const WriteLabel* lab)
{
	auto* wLab = llvm::dyn_cast<BInitWriteLabel>(lab);
	if (!wLab)
		return;

	/* Make sure the barrier hasn't already been initialized, and
	 * that the initializing value is greater than 0 */
	auto& g = getGraph();
	auto sIt = std::find_if(store_begin(g, wLab->getAddr()), store_end(g, wLab->getAddr()),
		[&g, wLab](auto& sLab) {
			return &sLab != wLab && sLab.getAddr() == wLab->getAddr() &&
				llvm::isa<BInitWriteLabel>(sLab);
		});

	if (sIt != store_end(g, wLab->getAddr()))
		reportError(wLab->getPos(), VerificationError::VE_InvalidBInit, "Called barrier_init() multiple times!", &*sIt);
	else if (wLab->getVal() == SVal(0))
		reportError(wLab->getPos(), VerificationError::VE_InvalidBInit, "Called barrier_init() with 0!");
	return;
}

void GenMCDriver::checkBIncValidity(const ReadLabel* rLab, const std::vector<Event>& rfs)
{
	auto* bLab = llvm::dyn_cast<BIncFaiReadLabel>(rLab);
	if (!bLab)
		return;

	if (std::any_of(rfs.cbegin(), rfs.cend(), [](const Event& rf) { return rf.isInitializer(); }))
		reportError(rLab->getPos(), VerificationError::VE_UninitializedMem,
			"Called barrier_wait() on uninitialized barrier!");
	else if (std::any_of(rfs.cbegin(), rfs.cend(), [this, bLab](const Event& rf) {
		auto rfVal = getWriteValue(getGraph().getEventLabel(rf), bLab->getAccess());
		return rfVal == SVal(0);
		}))
		reportError(rLab->getPos(), VerificationError::VE_AccessFreed,
			"Called barrier_wait() on destroyed barrier!", bLab->getRf());
}

void GenMCDriver::checkFinalAnnotations(const WriteLabel* wLab)
{
	if (!getConf()->helper)
		return;

	auto& g = getGraph();

	if (g.hasLocMoreThanOneStore(wLab->getAddr()))
		return;
	if ((wLab->isFinal() &&
		std::any_of(store_begin(g, wLab->getAddr()), store_end(g, wLab->getAddr()),
			[&](auto& sLab) { return !getHbView(wLab).contains(sLab.getPos()); })) ||
		(!wLab->isFinal() &&
			std::any_of(store_begin(g, wLab->getAddr()), store_end(g, wLab->getAddr()),
				[&](auto& sLab) { return sLab.isFinal(); }))) {
		reportError(wLab->getPos(), VerificationError::VE_Annotation,
			"Multiple stores at final location!");
		return;
	}
	return;
}

bool GenMCDriver::threadReadsMaximal(int tid)
{
	auto& g = getGraph();

	/*
	 * Depending on whether this is a DSA loop or not, we have to
	 * adjust the detection starting point: DSA-blocked threads
	 * will have a SpinStart as their last event.
	 */
	BUG_ON(!llvm::isa<BlockLabel>(g.getLastThreadLabel(tid)));
	auto* lastLab = g.getPreviousLabel(g.getLastThreadLabel(tid));
	auto start = llvm::isa<SpinStartLabel>(lastLab) ? lastLab->getPos().prev() : lastLab->getPos();
	for (auto j = start.index; j > 0; j--) {
		auto* lab = g.getEventLabel(Event(tid, j));
		BUG_ON(llvm::isa<LoopBeginLabel>(lab));
		if (llvm::isa<SpinStartLabel>(lab))
			return true;
		if (auto* rLab = llvm::dyn_cast<ReadLabel>(lab)) {
			if (!isCoMaximal(rLab->getAddr(), rLab->getRf()->getPos()))
				return false;
		}
	}
	BUG();
}

void GenMCDriver::checkLiveness()
{
	if (isHalting())
		return;

	const auto& g = getGraph();
	const auto* EE = getEE();

	/* Collect all threads blocked at spinloops */
	std::vector<int> spinBlocked;
	for (auto thrIt = EE->threads_begin(), thrE = EE->threads_end(); thrIt != thrE; ++thrIt) {
		auto* bLab = llvm::dyn_cast<BlockLabel>(g.getLastThreadLabel(thrIt->id));
		if (thrIt->getBlockageType() == BlockageType::Spinloop ||
			(bLab && bLab->getType() == BlockageType::Spinloop))
			spinBlocked.push_back(thrIt->id);
	}

	if (spinBlocked.empty())
		return;

	/* And check whether all of them are live or not */
	auto nonTermTID = 0u;
	if (std::all_of(spinBlocked.begin(), spinBlocked.end(), [&](int tid) {
		nonTermTID = tid;
		return threadReadsMaximal(tid);
		})) {
		/* Print some TID blocked by a spinloop */
		reportError(g.getLastThreadEvent(nonTermTID), VerificationError::VE_Liveness,
			"Non-terminating spinloop: thread " + std::to_string(nonTermTID));
	}
	return;
}

bool GenMCDriver::filterAcquiredLocks(const ReadLabel* rLab, std::vector<Event>& stores)

{
	auto& g = getGraph();

	/* The course of action depends on whether we are in repair mode or not */
	if ((llvm::isa<LockCasWriteLabel>(g.getEventLabel(stores.back())) ||
		llvm::isa<TrylockCasWriteLabel>(g.getEventLabel(stores.back()))) &&
		!isRescheduledRead(rLab->getPos())) {
		auto pos = rLab->getPos();
		g.removeLast(pos.thread);
		blockThread(pos, BlockageType::ReadOptBlock);
		return false;
	}

	auto max = stores.back();
	stores.erase(std::remove_if(stores.begin(), stores.end(), [&](const Event& s) {
		return (llvm::isa<LockCasWriteLabel>(g.getEventLabel(s)) ||
			llvm::isa<TrylockCasWriteLabel>(g.getEventLabel(s))) && s != max;
		}), stores.end());
	return true;
}

void GenMCDriver::filterConflictingBarriers(const ReadLabel* lab, std::vector<Event>& stores)
{
	if (getConf()->disableBAM || !llvm::isa<BIncFaiReadLabel>(lab))
		return;

	/* barrier_wait()'s FAI loads should not read from conflicting stores */
	auto& g = getGraph();
	stores.erase(std::remove_if(stores.begin(), stores.end(), [&](const Event& s) {
		return g.isStoreReadByExclusiveRead(s, lab->getAddr());
		}), stores.end());
	return;
}

int GenMCDriver::getSymmPredTid(int tid) const
{
	auto& g = getGraph();
	return g.getFirstThreadLabel(tid)->getSymmetricTid();
}

int GenMCDriver::getSymmSuccTid(int tid) const
{
	auto& g = getGraph();
	auto symm = tid;

	/* Check if there is anyone else symmetric to SYMM */
	for (auto i = tid + 1; i < g.getNumThreads(); i++)
		if (g.getFirstThreadLabel(i)->getSymmetricTid() == symm)
			return i;
	return -1; /* no one else */
}

bool GenMCDriver::isEcoBefore(const EventLabel* lab, int tid) const
{
	auto& g = getGraph();
	if (!llvm::isa<MemAccessLabel>(lab))
		return false;

	auto symmPos = Event(tid, lab->getIndex());
	// if (auto *wLab = rf_pred(g, lab); wLab) {
	// 	return wLab.getPos() == symmPos;
	// }))
	// 	return true;
	if (std::any_of(co_succ_begin(g, lab), co_succ_end(g, lab), [&](auto& sLab) {
		return sLab.getPos() == symmPos ||
			std::any_of(sLab.readers_begin(), sLab.readers_end(), [&](auto& rLab) {
			return rLab.getPos() == symmPos;
				});
		}))
		return true;
	if (std::any_of(fr_succ_begin(g, lab), fr_succ_end(g, lab), [&](auto& sLab) {
		return sLab.getPos() == symmPos ||
			std::any_of(sLab.readers_begin(), sLab.readers_end(), [&](auto& rLab) {
			return rLab.getPos() == symmPos;
				});
		}))
		return true;
	return false;
}

bool GenMCDriver::isEcoSymmetric(const EventLabel* lab, int tid) const
{
	auto& g = getGraph();

	auto* symmLab = g.getEventLabel(Event(tid, lab->getIndex()));
	if (auto* rLab = llvm::dyn_cast<ReadLabel>(lab)) {
		return rLab->getRf() == llvm::dyn_cast<ReadLabel>(symmLab)->getRf();
	}

	auto* wLab = llvm::dyn_cast<WriteLabel>(lab);
	BUG_ON(!wLab);
	return g.co_imm_succ(wLab) == llvm::dyn_cast<WriteLabel>(symmLab);
}

bool GenMCDriver::isPredSymmetryOK(const EventLabel* lab, int symm)
{
	auto& g = getGraph();

	BUG_ON(symm == -1);
	if (!sharePrefixSR(symm, lab->getPos()) || !g.containsPos(Event(symm, lab->getIndex())))
		return true;

	auto* symmLab = g.getEventLabel(Event(symm, lab->getIndex()));
	if (symmLab->getKind() != lab->getKind())
		return true;

	return !isEcoBefore(lab, symm);
}

bool GenMCDriver::isPredSymmetryOK(const EventLabel* lab)
{
	auto& g = getGraph();
	std::vector<int> preds;

	auto symm = getSymmPredTid(lab->getThread());
	while (symm != -1) {
		preds.push_back(symm);
		symm = getSymmPredTid(symm);
	}
	return std::all_of(preds.begin(), preds.end(), [&](auto& symm) { return isPredSymmetryOK(lab, symm); });
}

bool GenMCDriver::isSuccSymmetryOK(const EventLabel* lab, int symm)
{
	auto& g = getGraph();

	BUG_ON(symm == -1);
	if (!sharePrefixSR(symm, lab->getPos()) || !g.containsPos(Event(symm, lab->getIndex())))
		return true;

	auto* symmLab = g.getEventLabel(Event(symm, lab->getIndex()));
	if (symmLab->getKind() != lab->getKind())
		return true;

	return !isEcoBefore(symmLab, lab->getThread());
}

bool GenMCDriver::isSuccSymmetryOK(const EventLabel* lab)
{
	auto& g = getGraph();
	std::vector<int> succs;

	auto symm = getSymmSuccTid(lab->getThread());
	while (symm != -1) {
		succs.push_back(symm);
		symm = getSymmSuccTid(symm);
	}
	return std::all_of(succs.begin(), succs.end(), [&](auto& symm) { return isSuccSymmetryOK(lab, symm); });
}

bool GenMCDriver::isSymmetryOK(const EventLabel* lab)
{
	auto& g = getGraph();
	return isPredSymmetryOK(lab) && isSuccSymmetryOK(lab);
}

void GenMCDriver::updatePrefixWithSymmetriesSR(EventLabel* lab)
{
	auto t = getSymmPredTid(lab->getThread());
	if (t == -1)
		return;

	auto& v = lab->getPrefixView();
	auto si = calcLargestSymmPrefixBeforeSR(t, lab->getPos());
	auto* symmLab = getGraph().getEventLabel({ t, si });
	v.update(getPrefixView(symmLab));
	if (auto* rLab = llvm::dyn_cast<ReadLabel>(symmLab)) {
		v.update(getPrefixView(rLab->getRf()));
	}
}

int GenMCDriver::calcLargestSymmPrefixBeforeSR(int tid, Event pos) const
{
	auto& g = getGraph();

	if (tid < 0 || tid >= g.getNumThreads())
		return -1;

	auto limit = std::min((long)pos.index, (long)g.getThreadSize(tid) - 1);
	for (auto j = 0; j < limit; j++) {
		auto* labA = g.getEventLabel(Event(tid, j));
		auto* labB = g.getEventLabel(Event(pos.thread, j));

		if (labA->getKind() != labB->getKind())
			return j - 1;
		if (auto* rLabA = llvm::dyn_cast<ReadLabel>(labA)) {
			auto* rLabB = llvm::dyn_cast<ReadLabel>(labB);
			if (rLabA->getRf()->getThread() == tid &&
				rLabB->getRf()->getThread() == pos.thread &&
				rLabA->getRf()->getIndex() == rLabB->getRf()->getIndex())
				continue;
			if (rLabA->getRf() != rLabB->getRf())
				return j - 1;
		}
		if (auto* wLabA = llvm::dyn_cast<WriteLabel>(labA))
			if (!wLabA->isLocal())
				return j - 1;
	}
	return limit;
}

bool GenMCDriver::sharePrefixSR(int tid, Event pos) const
{
	return calcLargestSymmPrefixBeforeSR(tid, pos) == pos.index;
}

void GenMCDriver::filterSymmetricStoresSR(const ReadLabel* rLab, std::vector<Event>& stores) const
{
	auto& g = getGraph();
	auto* EE = getEE();
	auto t = getSymmPredTid(rLab->getThread());

	/* If there is no symmetric thread, exit */
	if (t == -1)
		return;

	/* Check whether the po-prefixes of the two threads match */
	if (!sharePrefixSR(t, rLab->getPos()))
		return;

	/* Get the symmetric event and make sure it matches as well */
	auto* lab = llvm::dyn_cast<ReadLabel>(g.getEventLabel(Event(t, rLab->getIndex())));
	if (!lab || lab->getAddr() != rLab->getAddr() || lab->getSize() != lab->getSize())
		return;

	if (!g.isRMWLoad(lab))
		return;

	/* Remove stores that will be explored symmetrically */
	auto rfStamp = lab->getRf()->getStamp();
	stores.erase(std::remove_if(stores.begin(), stores.end(), [&](auto s) {
		return lab->getRf()->getPos() == s;
		}), stores.end());
	return;
}

bool GenMCDriver::filterValuesFromAnnotSAVER(const ReadLabel* rLab, std::vector<Event>& validStores)
{
	if (!rLab->getAnnot())
		return false;

	using Evaluator = SExprEvaluator<ModuleID::ID>;

	auto& g = getGraph();

	/* Ensure we keep the maximal store around even if Helper messed with it */
	BUG_ON(validStores.empty());
	auto maximal = validStores.back();
	validStores.erase(std::remove_if(validStores.begin(), validStores.end(), [&](Event w) {
		auto val = getWriteValue(g.getEventLabel(w), rLab->getAccess());
		return w != maximal && !isCoMaximal(rLab->getAddr(), w, true) &&
			!Evaluator().evaluate(rLab->getAnnot(), val);
		}), validStores.end());
	BUG_ON(validStores.empty());

	/* Return whether we should block */
	auto maximalVal = getWriteValue(g.getEventLabel(validStores.back()), rLab->getAccess());
	return !Evaluator().evaluate(rLab->getAnnot(), maximalVal);
}

void GenMCDriver::unblockWaitingHelping()
{
	/* We have to wake up all threads waiting on helping CASes,
	 * as we don't know which ones are from the same CAS */
	std::for_each(getEE()->threads_begin(), getEE()->threads_end(), [](Thread& thr) {
		if (thr.isBlocked() && thr.getBlockageType() == BlockageType::HelpedCas)
			thr.unblock();
		});
	for (auto i = 0u; i < getGraph().getNumThreads(); i++) {
		auto* bLab = llvm::dyn_cast<BlockLabel>(getGraph().getLastThreadLabel(i));
		if (bLab && bLab->getType() == BlockageType::HelpedCas)
			getGraph().removeLast(bLab->getThread());
	}
}

bool GenMCDriver::writesBeforeHelpedContainedInView(const HelpedCasReadLabel* lab, const View& view)
{
	auto& g = getGraph();
	auto& hb = getHbView(lab);

	for (auto i = 0u; i < hb.size(); i++) {
		auto j = hb.getMax(i);
		while (!llvm::isa<WriteLabel>(g.getEventLabel(Event(i, j))) && j > 0)
			--j;
		if (j > 0 && !view.contains(Event(i, j)))
			return false;
	}
	return true;
}

bool GenMCDriver::checkHelpingCasCondition(const HelpingCasLabel* hLab)
{
	auto& g = getGraph();
	auto* EE = getEE();

	auto hs = g.collectAllEvents([&](const EventLabel* lab) {
		auto* rLab = llvm::dyn_cast<HelpedCasReadLabel>(lab);
		return rLab && g.isRMWLoad(rLab) && rLab->getAddr() == hLab->getAddr() &&
			rLab->getType() == hLab->getType() && rLab->getSize() == hLab->getSize() &&
			rLab->getOrdering() == hLab->getOrdering() &&
			rLab->getExpected() == hLab->getExpected() &&
			rLab->getSwapVal() == hLab->getSwapVal();
		});

	if (hs.empty())
		return false;

	if (std::any_of(hs.begin(), hs.end(), [&g, EE, this](const Event& h) {
		auto* hLab = llvm::dyn_cast<HelpedCasReadLabel>(g.getEventLabel(h));
		auto& view = getHbView(hLab);
		return !writesBeforeHelpedContainedInView(hLab, view);
		}))
		ERROR("Helped/Helping CAS annotation error! "
			"Not all stores before helped-CAS are visible to helping-CAS!\n");
	return true;
}

bool GenMCDriver::checkAtomicity(const WriteLabel* wLab)
{
	if (getGraph().violatesAtomicity(wLab)) {
		moot();
		return false;
	}
	return true;
}

bool GenMCDriver::ensureConsistentRf(const ReadLabel* rLab, std::vector<Event>& rfs)
{
	auto& g = getGraph();

	rfs.erase(std::remove_if(rfs.begin(), rfs.end(), [&](auto& rf) {
		g.changeRf(rLab->getPos(), rf);
		return !isExecutionValid(rLab);
		}), rfs.end());

	if (rfs.empty()) {
		getEE()->block(BlockageType::Cons);
		return false;
	}
	g.changeRf(rLab->getPos(), rfs.back());
	return true;
}

bool GenMCDriver::ensureConsistentStore(const WriteLabel* wLab)
{
	if (!checkAtomicity(wLab) || !isExecutionValid(wLab)) {
		getEE()->block(BlockageType::Cons);
		moot();
		return false;
	}
	return true;
}

void GenMCDriver::filterInvalidRecRfs(const ReadLabel* rLab, std::vector<Event>& rfs)
{
	auto& g = getGraph();
	rfs.erase(std::remove_if(rfs.begin(), rfs.end(), [&](Event& r) {
		g.changeRf(rLab->getPos(), r);
		return !isRecoveryValid(rLab);
		}), rfs.end());
	BUG_ON(rfs.empty());
	g.changeRf(rLab->getPos(), rfs[0]);
	return;
}

void GenMCDriver::handleThreadKill(std::unique_ptr<ThreadKillLabel> kLab)
{
	BUG_ON(isExecutionDrivenByGraph(&*kLab));
	addLabelToGraph(std::move(kLab));
	return;
}

bool GenMCDriver::isSymmetricToSR(int candidate, Event parent, const ThreadInfo& info) const
{
	auto& g = getGraph();
	auto cParent = g.getFirstThreadLabel(candidate)->getParentCreate();
	auto& cInfo = g.getFirstThreadLabel(candidate)->getThreadInfo();

	/* A tip to print to the user in case two threads look
	 * symmetric, but we cannot deem it */
	auto tipSymmetry = [&]() {
		LOG_ONCE("possible-symmetry", VerbosityLevel::Tip)
			<< "Threads (" << getEE()->getThrById(cInfo.id)
			<< ") and (" << getEE()->getThrById(info.id)
			<< ") could benefit from symmetry reduction."
			<< " Consider using __VERIFIER_spawn_symmetric().\n";
		};

	/* First, check that the two threads are actually similar */
	if (cInfo.id == info.id ||
		cInfo.parentId != info.parentId ||
		cInfo.funId != info.funId ||
		cInfo.arg != info.arg) {
		if (cInfo.funId == info.funId && cInfo.parentId == info.parentId)
			tipSymmetry();
		return false;
	}

	/* Then make sure that there is no memory access in between the spawn events */
	auto mm = std::minmax(parent.index, cParent.index);
	auto minI = mm.first;
	auto maxI = mm.second;
	for (auto j = minI; j < maxI; j++) {
		if (llvm::isa<MemAccessLabel>(g.getEventLabel(Event(parent.thread, j)))) {
			tipSymmetry();
			return false;
		}
	}
	return true;
}

int GenMCDriver::getSymmetricTidSR(const ThreadCreateLabel* tcLab, const ThreadInfo& childInfo) const
{
	if (!getConf()->symmetryReduction)
		return -1;

	/* Has the user provided any info? */
	if (childInfo.symmId != -1)
		return childInfo.symmId;

	auto& g = getGraph();
	auto* EE = getEE();

	for (auto i = childInfo.id - 1; i > 0; i--)
		if (isSymmetricToSR(i, tcLab->getPos(), childInfo))
			return i;
	return -1;
}

int GenMCDriver::handleThreadCreate(std::unique_ptr<ThreadCreateLabel> tcLab)
{
	auto& g = getGraph();
	auto* EE = getEE();

	if (isExecutionDrivenByGraph(&*tcLab))
		return llvm::dyn_cast<ThreadCreateLabel>(g.getEventLabel(tcLab->getPos()))->getChildId();

	/* First, check if the thread to be created already exists */
	int cid = 0;
	while (cid < (long)g.getNumThreads()) {
		if (!g.isThreadEmpty(cid)) {
			auto* bLab = llvm::dyn_cast<ThreadStartLabel>(g.getFirstThreadLabel(cid));
			BUG_ON(!bLab);
			if (bLab->getParentCreate() == tcLab->getPos())
				break;
		}
		++cid;
	}

	/* Add an event for the thread creation */
	tcLab->setChildId(cid);
	auto* lab = llvm::dyn_cast<ThreadCreateLabel>(addLabelToGraph(std::move(tcLab)));

	/* Prepare the execution context for the new thread */
	EE->constructAddThreadFromInfo(lab->getChildInfo());

	/* If the thread does not exist in the graph, make an entry for it */
	if (cid == (long)g.getNumThreads()) {
		g.addNewThread();
		BUG_ON(EE->getNumThreads() != g.getNumThreads());
	}
	else {
		BUG_ON(g.getThreadSize(cid) != 1);
		g.removeLast(cid);
	}
	auto symm = getSymmetricTidSR(lab, lab->getChildInfo());
	auto tsLab = ThreadStartLabel::create(Event(cid, 0), lab->getPos(), lab->getChildInfo(), symm);
	addLabelToGraph(std::move(tsLab));
	return cid;
}

std::optional<SVal>
GenMCDriver::handleThreadJoin(std::unique_ptr<ThreadJoinLabel> lab)
{
	Print(RED("handle thread join"), lab->getPos());

	auto& g = getGraph();
	auto& thr = getEE()->getCurThr();

	if (isExecutionDrivenByGraph(&*lab))
		return { getJoinValue(llvm::dyn_cast<ThreadJoinLabel>(g.getEventLabel(lab->getPos()))) };

	if (!llvm::isa<ThreadFinishLabel>(g.getLastThreadLabel(lab->getChildId()))) {
		blockThread(lab->getPos(), BlockageType::ThreadJoin);
		return std::nullopt;
	}

	auto* jLab = llvm::dyn_cast<ThreadJoinLabel>(addLabelToGraph(std::move(lab)));
	auto cid = jLab->getChildId();

	auto* eLab = llvm::dyn_cast<ThreadFinishLabel>(g.getLastThreadLabel(cid));
	BUG_ON(!eLab);
	eLab->setParentJoin(jLab);

	if (cid < 0 || long(g.getNumThreads()) <= cid || cid == thr.id) {
		std::string err = "ERROR: Invalid TID in pthread_join(): " + std::to_string(cid);
		if (cid == thr.id)
			err += " (TID cannot be the same as the calling thread)";
		reportError(jLab->getPos(), VerificationError::VE_InvalidJoin, err);
		return { SVal(0) };
	}
	return { getJoinValue(jLab) };
}

void GenMCDriver::handleThreadFinish(std::unique_ptr<ThreadFinishLabel> eLab)
{
	auto& g = getGraph();
	auto* EE = getEE();
	auto& thr = EE->getCurThr();

	if (!isExecutionDrivenByGraph(&*eLab) && /* Make sure that there is not a failed assume... */
		!thr.isBlocked()) {
		auto* lab = addLabelToGraph(std::move(eLab));

		if (thr.id == 0)
			return;

		for (auto i = 0u; i < g.getNumThreads(); i++) {
			auto* pLab = llvm::dyn_cast<BlockLabel>(g.getLastThreadLabel(i));
			if (pLab && pLab->getType() == BlockageType::ThreadJoin) {
				/* If parent thread is waiting for me, relieve it.
				 * We do not keep track of who is waiting for whom now,
				 * so just unblock everyone. */
				unblockThread(pLab->getPos());
			}
		}
	}
}

void GenMCDriver::handleFenceLKMM(std::unique_ptr<FenceLabel> fLab)
{
	if (isExecutionDrivenByGraph(&*fLab))
		return;

	addLabelToGraph(std::move(fLab));
	return;
}

void GenMCDriver::handleFence(std::unique_ptr<FenceLabel> fLab)
{
	if (llvm::isa<SmpFenceLabelLKMM>(&*fLab)) {
		handleFenceLKMM(std::move(fLab));
		return;
	}

	if (isExecutionDrivenByGraph(&*fLab))
		return;

	addLabelToGraph(std::move(fLab));
	return;
}

void GenMCDriver::handleCLFlush(std::unique_ptr<CLFlushLabel> fLab)
{
	if (isExecutionDrivenByGraph(&*fLab))
		return;

	addLabelToGraph(std::move(fLab));
	return;
}

void GenMCDriver::checkReconsiderFaiSpinloop(const MemAccessLabel* lab)
{
	auto& g = getGraph();
	auto* EE = getEE();

	for (auto i = 0u; i < g.getNumThreads(); i++) {
		auto& thr = EE->getThrById(i);

		/* Is there any thread blocked on a potential spinloop? */
		auto* eLab = llvm::dyn_cast<BlockLabel>(g.getLastThreadLabel(i));
		if (!eLab || eLab->getType() != BlockageType::FaiZNESpinloop)
			continue;

		/* Check whether this access affects the spinloop variable */
		auto* faiLab = llvm::dyn_cast<FaiWriteLabel>(g.getPreviousLabelST(eLab,
			[](const EventLabel* lab) { return llvm::isa<FaiWriteLabel>(lab); }));
		if (faiLab->getAddr() != lab->getAddr())
			continue;
		/* FAIs on the same variable are OK... */
		if (llvm::isa<FaiReadLabel>(lab) || llvm::isa<FaiWriteLabel>(lab))
			continue;

		/* If it does, and also breaks the assumptions, unblock thread */
		if (!getHbView(faiLab).contains(lab->getPos())) {
			auto pos = eLab->getPos();
			unblockThread(pos);
			addLabelToGraph(FaiZNESpinEndLabel::create(pos));
		}
	}
	return;
}

std::vector<Event> GenMCDriver::getRfsApproximation(const ReadLabel* lab)
{
	auto rfs = getCoherentStores(lab->getAddr(), lab->getPos());
	if (!llvm::isa<CasReadLabel>(lab) && !llvm::isa<FaiReadLabel>(lab)) {

		return rfs;
	}
	// Print(format(rfs), "containing cas or fai, removing some stores");
	// getchar();

	/* Remove atomicity violations */
	auto& g = getGraph();
	auto& before = getPrefixView(lab);
	// Print("repfix view of read:", before);
	rfs.erase(std::remove_if(rfs.begin(), rfs.end(), [&](const Event& s) {
		auto oldVal = getWriteValue(g.getEventLabel(s), lab->getAccess());
		if (llvm::isa<FaiReadLabel>(lab) && g.isStoreReadBySettledRMW(s, lab->getAddr(), before))
			return true;
		if (auto* rLab = llvm::dyn_cast<CasReadLabel>(lab)) {
			if (oldVal == rLab->getExpected() &&
				g.isStoreReadBySettledRMW(s, rLab->getAddr(), before))
				return true;
		}
		return false;
		}), rfs.end());
	return rfs;
}

void GenMCDriver::filterConfirmingRfs(const ReadLabel* lab, std::vector<Event>& stores)
{
	auto& g = getGraph();
	if (!getConf()->helper || !g.isConfirming(lab))
		return;

	auto sc = Event::getInit();
	auto* rLab = llvm::dyn_cast<ReadLabel>(
		g.getEventLabel(g.getMatchingSpeculativeRead(lab->getPos(), &sc)));
	ERROR_ON(!rLab, "Confirming annotation error! Does the speculative "
		"read always precede the confirming operation?\n");

	/* Print a warning if there are ABAs */
	auto specVal = getWriteValue(rLab->getRf(), rLab->getAccess());
	auto valid = std::count_if(stores.begin(), stores.end(), [&](const Event& s) {
		return getWriteValue(g.getEventLabel(s), rLab->getAccess()) == specVal;
		});
	WARN_ON_ONCE(valid > 1, "helper-aba-found",
		"Possible ABA pattern on variable " + getVarName(rLab->getAddr()) +
		"! Consider running without -helper.\n");

	/* Do not optimize if there are intervening SC accesses */
	if (!sc.isInitializer())
		return;

	BUG_ON(stores.empty());

	/* Demand that the confirming read reads the speculated value (exact rf) */
	auto maximal = stores.back();
	stores.erase(std::remove_if(stores.begin(), stores.end(), [&](const Event& s) {
		return s != rLab->getRf()->getPos();
		}), stores.end());

	/* ... and if no such value exists, block indefinitely */
	if (stores.empty()) {
		stores.push_back(maximal);
		blockThreadTryMoot(lab->getPos().next(), BlockageType::Confirmation);
		return;
	}

	/* deprioritize thread upon confirmation */
	if (!threadPrios.empty() &&
		llvm::isa<SpeculativeReadLabel>(getGraph().getEventLabel(threadPrios[0])))
		threadPrios.clear();
	return;
}

bool GenMCDriver::existsPendingSpeculation(const ReadLabel* lab, const std::vector<Event>& stores)
{
	auto& g = getGraph();
	return (std::any_of(label_begin(g), label_end(g), [&](auto& oLab) {
		auto* orLab = llvm::dyn_cast<SpeculativeReadLabel>(&oLab);
		return orLab && orLab->getAddr() == lab->getAddr() &&
			!getHbView(lab).contains(orLab->getPos()) &&
			orLab->getPos() != lab->getPos();
		}) &&
		std::find_if(stores.begin(), stores.end(), [&](const Event& s) {
			return llvm::isa<ConfirmingCasWriteLabel>(g.getEventLabel(s)) &&
				s.index > lab->getIndex() && s.thread == lab->getThread();
			}) == stores.end());
}

bool GenMCDriver::filterUnconfirmedReads(const ReadLabel* lab, std::vector<Event>& stores)
{
	if (!getConf()->helper || !llvm::isa<SpeculativeReadLabel>(lab))
		return true;

	if (isRescheduledRead(lab->getPos())) {
		setRescheduledRead(Event::getInit());
		return true;
	}

	/* If there exist any speculative reads the confirming read of which has not been added,
	 * prioritize those and discard current rfs; otherwise, prioritize ourselves */
	if (existsPendingSpeculation(lab, stores)) {
		std::swap(stores[0], stores.back());
		stores.resize(1);
		auto pos = lab->getPos();
		getGraph().removeLast(pos.thread);
		blockThread(pos, BlockageType::ReadOptBlock);
		return false;
	}

	threadPrios = { lab->getPos() };
	return true;
}

bool GenMCDriver::filterOptimizeRfs(const ReadLabel* lab, std::vector<Event>& stores)
{
	/* Symmetry reduction */
	if (getConf()->symmetryReduction)
		filterSymmetricStoresSR(lab, stores);

	/* BAM */
	if (!getConf()->disableBAM)
		filterConflictingBarriers(lab, stores);

	/* Locking */
	if (llvm::isa<LockCasReadLabel>(lab))
		if (!filterAcquiredLocks(lab, stores))
			return false;

	/* Helper: Try to read speculated value (affects maximality status) */
	if (getConf()->helper && getGraph().isConfirming(lab))
		filterConfirmingRfs(lab, stores);

	/* Helper: If there are pending confirmations, prioritize those */
	if (getConf()->helper && llvm::isa<SpeculativeReadLabel>(lab))
		if (!filterUnconfirmedReads(lab, stores))
			return false;

	/* If this load is annotatable, keep values that will not leed to blocking */
	if (lab->getAnnot() && !inEstimationMode())
		filterValuesFromAnnotSAVER(lab, stores);
	return true;
}

void GenMCDriver::filterAtomicityViolations(const ReadLabel* rLab, std::vector<Event>& stores)
{
	auto& g = getGraph();
	if (!llvm::isa<CasReadLabel>(rLab) && !llvm::isa<FaiReadLabel>(rLab))
		return;

	const auto* casLab = llvm::dyn_cast<CasReadLabel>(rLab);
	auto valueMakesSuccessfulRMW = [&casLab, rLab](auto&& val) { return !casLab || val == casLab->getExpected(); };
	stores.erase(std::remove_if(stores.begin(), stores.end(), [&](auto& s) {
		auto* sLab = g.getEventLabel(s);
		if (auto* iLab = llvm::dyn_cast<InitLabel>(sLab))
			return std::any_of(iLab->rf_begin(rLab->getAddr()), iLab->rf_end(rLab->getAddr()), [&](auto& rLab) {
			return g.isRMWLoad(&rLab) && valueMakesSuccessfulRMW(getReadValue(&rLab));
				});
		return std::any_of(rf_succ_begin(g, sLab), rf_succ_end(g, sLab), [&](auto& rLab) {
			return g.isRMWLoad(&rLab) && valueMakesSuccessfulRMW(getReadValue(&rLab));
			});
		}), stores.end());
}

void GenMCDriver::updateStSpaceChoices(const ReadLabel* rLab, const std::vector<Event>& stores)
{
	auto& choices = getChoiceMap();
	choices[rLab->getStamp()] = stores;
}

std::optional<SVal> GenMCDriver::pickRandomRf(ReadLabel* rLab, const std::vector<Event>& stores)
{
	auto& g = getGraph();

	MyDist dist(0, stores.size() - 1);
	auto random = dist(estRng);
	g.changeRf(rLab->getPos(), stores[random]);

	if (readsUninitializedMem(rLab)) {
		reportError(rLab->getPos(), VerificationError::VE_UninitializedMem);
		return std::nullopt;
	}
	return getWriteValue(rLab->getRf(), rLab->getAccess());
}

std::optional<SVal>
GenMCDriver::handleLoad(std::unique_ptr<ReadLabel> rLab)
{
#ifdef DEBUG_LUAN
	if (rLab) {
		Print("\nhandleLoad:", rLab->getPos());
	}
	Print("coherence:");
	// PrintCoherence(getGraph());
	// Print("recovery ?", inRecoveryMode());
	Print("driven by graph (already there in graph) ?", isExecutionDrivenByGraph(&*rLab) ? "true" : "false");
	// getchar();

#endif
	auto& g = getGraph();
	auto* EE = getEE();
	auto& thr = EE->getCurThr();

	if (inRecoveryMode() && rLab->getAddr().isVolatile())
		return { getRecReadRetValue(rLab.get()) };

	if (isExecutionDrivenByGraph(&*rLab)) {
#ifdef FUZZ_BACKWARD
		EventLabel* eLab = g.getEventLabel(rLab->getPos());
		if (auto rlab = llvm::dyn_cast<ReadLabel>(eLab); !rlab) {
			Print(RED("rLab would be nullptr, cann't get value"));
			Print("rLab->getPos() =", rLab->getPos());
			Print("eLab =", *eLab);
			// getchar();
		}
#endif


		return { getReadRetValueAndMaybeBlock(llvm::dyn_cast<ReadLabel>(g.getEventLabel(rLab->getPos()))) };
	}


	/* First, we have to check whether the access is valid. This has to
	 * happen here because we may query the interpreter for this location's
	 * value in order to determine whether this load is going to be an RMW.
	 * Coherence needs to be tracked before validity is established, as
	 * consistency checks may be triggered if the access is invalid */
	g.trackCoherenceAtLoc(rLab->getAddr());

	if (!rLab->getAnnot())
		rLab->setAnnot(EE->getCurrentAnnotConcretized());
	cacheEventLabel(&*rLab);
	auto* lab = llvm::dyn_cast<ReadLabel>(addLabelToGraph(std::move(rLab)));

	if (!isAccessValid(lab)) {
		reportError(lab->getPos(), VerificationError::VE_AccessNonMalloc);
		return std::nullopt; /* This execution will be blocked */
	}
	g.addAlloc(findAllocatingLabel(g, lab->getAddr()), lab);

	if (checkForRaces(lab) != VerificationError::VE_OK)
		return std::nullopt;

	/* Get an approximation of the stores we can read from */
	auto stores = getRfsApproximation(lab);
	BUG_ON(stores.empty());
	GENMC_DEBUG(LOG(VerbosityLevel::Debug3) << "Rfs: " << format(stores) << "\n"; );



	/* Try to minimize the number of rfs */
	if (!filterOptimizeRfs(lab, stores))
		return std::nullopt;
#ifdef DEBUG_LUAN
	// std::shuffle(stores.begin(), stores.end(), rng);	// don't do this
	Print("Rfs:", format(stores));
#endif

	/* ... add an appropriate label with a random rf */
	g.changeRf(lab->getPos(), stores.back());
	GENMC_DEBUG(LOG(VerbosityLevel::Debug3) << "Rfs (optimized): " << format(stores) << "\n"; );

	/* ... and make sure that the rf we end up with is consistent */
	if (!ensureConsistentRf(lab, stores))
		return std::nullopt;

	if (readsUninitializedMem(lab)) {
		reportError(lab->getPos(), VerificationError::VE_UninitializedMem);
		return std::nullopt;
	}

	/* If this is the last part of barrier_wait() check whether we should block */
	auto retVal = getWriteValue(g.getEventLabel(stores.back()), lab->getAccess());
	if (llvm::isa<BWaitReadLabel>(lab) &&
		retVal != getBarrierInitValue(lab->getAccess())) {
		if (!getConf()->disableBAM) {
			auto pos = lab->getPos();
			g.removeLast(pos.thread);
			blockThread(pos, BlockageType::Barrier);
			return { retVal };
		}
		getEE()->getCurThr().block(BlockageType::Barrier);
	}

	if (isRescheduledRead(lab->getPos()))
		setRescheduledRead(Event::getInit());

	if (inEstimationMode()) {	// currect to do
		updateStSpaceChoices(lab, stores);
		filterAtomicityViolations(lab, stores);
		return pickRandomRf(lab, stores);
	}

	GENMC_DEBUG(
		LOG(VerbosityLevel::Debug2)
		<< "--- Added load " << lab->getPos() << "\n" << getGraph();
	);

	/* Check whether the load forces us to reconsider some existing event */
	checkReconsiderFaiSpinloop(lab);

	/* Check for races and reading from uninitialized memory */
	if (llvm::isa<LockCasReadLabel>(lab))
		checkLockValidity(lab, stores);
	if (llvm::isa<BIncFaiReadLabel>(lab))
		checkBIncValidity(lab, stores);

	/* Push all the other alternatives choices to the Stack (many maximals for wb) */
	std::for_each(stores.begin(), stores.end() - 1, [&](const Event& s) {
		auto status = false; /* MO messes with the status */
		addToWorklist(lab->getStamp(), std::make_unique<ReadForwardRevisit>(lab->getPos(), s, status));
		});
	return { retVal };
}

void GenMCDriver::annotateStoreHELPER(WriteLabel* wLab)
{
	auto& g = getGraph();

	/* Don't bother with lock ops */
	if (!getConf()->helper || !g.isRMWStore(wLab) || llvm::isa<LockCasWriteLabel>(wLab) ||
		llvm::isa<TrylockCasWriteLabel>(wLab))
		return;

	/* Check whether we can mark it as RevBlocker */
	auto* pLab = g.getPreviousLabel(wLab);
	auto* mLab = llvm::dyn_cast_or_null<MemAccessLabel>(
		getPreviousVisibleAccessLabel(pLab->getPos()));
	auto* rLab = llvm::dyn_cast_or_null<ReadLabel>(mLab);
	if (!mLab || (mLab->wasAddedMax() && (!rLab || rLab->isRevisitable())))
		return;

	/* Mark the store and its predecessor */
	if (llvm::isa<FaiWriteLabel>(wLab))
		llvm::dyn_cast<FaiReadLabel>(pLab)->setAttr(WriteAttr::RevBlocker);
	else
		llvm::dyn_cast<CasReadLabel>(pLab)->setAttr(WriteAttr::RevBlocker);
	wLab->setAttr(WriteAttr::RevBlocker);
}

std::vector<Event> GenMCDriver::getRevisitableApproximation(const WriteLabel* sLab)
{
	auto& g = getGraph();
	auto& prefix = getPrefixView(sLab);
	auto loads = getCoherentRevisits(sLab, prefix);
	std::sort(loads.begin(), loads.end(), [&g](const Event& l1, const Event& l2) {
		return g.getEventLabel(l1)->getStamp() > g.getEventLabel(l2)->getStamp();
		});
	return loads;
}

void GenMCDriver::pickRandomCo(WriteLabel* sLab,
	const llvm::iterator_range<ExecutionGraph::co_iterator>& placesRange)
{
	auto& g = getGraph();

	MyDist dist(0, std::distance(placesRange.begin(), placesRange.end()));
	auto random = dist(estRng);
	g.addStoreToCO(sLab, std::next(placesRange.begin(), (long long)random));
}

void GenMCDriver::updateStSpaceChoices(const WriteLabel* wLab, const std::vector<Event>& stores)
{
	auto& choices = getChoiceMap();
	choices[wLab->getStamp()] = stores;
}

void GenMCDriver::calcCoOrderings(WriteLabel* lab)
{
	/* Find all possible placings in coherence for this store */
	auto& g = getGraph();
	auto placesRange = getCoherentPlacings(lab->getAddr(), lab->getPos(), g.isRMWStore(lab));

	if (inEstimationMode()) {
		std::vector<Event> cos;
		std::transform(placesRange.begin(), placesRange.end(), std::back_inserter(cos), [&](auto& lab) { return lab.getPos(); });
		cos.push_back(Event::getBottom());
		pickRandomCo(lab, placesRange);
		updateStSpaceChoices(lab, cos);
		return;
	}

	/* We cannot place the write just before the write of an RMW or during recovery */
	for (auto& succLab : placesRange) {
		if (!g.isRMWStore(succLab.getPos()) && !inRecoveryMode())
			addToWorklist(lab->getStamp(),
				std::make_unique<WriteForwardRevisit>(lab->getPos(), succLab.getPos()));
	}
	g.addStoreToCO(lab, placesRange.end());
}

void GenMCDriver::handleStore(std::unique_ptr<WriteLabel> wLab)
{
#ifdef DEBUG_LUAN
	// if (wLab) {
	// 	Print("\nhandleStore:", wLab->getPos(), "at", wLab->getAddr());
	// 	Print("before handling store");
	// 	PrintCoherence(getGraph());

	// 	if (isExecutionDrivenByGraph(&*wLab)) {
	// 		Print("IS driven by graph");
	// 	}
	// 	else {
	// 		Print("NOT driven by graph");
	// 	}
	// }

#endif
	if (isExecutionDrivenByGraph(&*wLab))
		return;

	auto& g = getGraph();
	auto* EE = getEE();

	/* If it's a valid access, track coherence for this location */
	g.trackCoherenceAtLoc(wLab->getAddr());

	if (getConf()->helper && g.isRMWStore(&*wLab))
		annotateStoreHELPER(&*wLab);

	auto* lab = llvm::dyn_cast<WriteLabel>(addLabelToGraph(std::move(wLab)));

	if (!isAccessValid(lab)) {
		reportError(lab->getPos(), VerificationError::VE_AccessNonMalloc);
		return;
	}
	g.addAlloc(findAllocatingLabel(g, lab->getAddr()), lab);

	/* It is always consistent to add the store at the end of MO */
	if (llvm::isa<BIncFaiWriteLabel>(lab) && lab->getVal() == SVal(0))
		lab->setVal(getBarrierInitValue(lab->getAccess()));

	calcCoOrderings(lab);

	/* If the graph is not consistent (e.g., w/ LAPOR) stop the exploration */
	bool cons = ensureConsistentStore(lab);

	GENMC_DEBUG(LOG(VerbosityLevel::Debug2)
		<< "--- Added store " << lab->getPos() << "\n" << getGraph(); );

	if (cons && checkForRaces(lab) != VerificationError::VE_OK)
		return;

	if (!inRecoveryMode() && !inReplay())
		calcRevisits(lab);

	if (!cons)
		return;

	checkReconsiderFaiSpinloop(lab);
	if (llvm::isa<HelpedCasWriteLabel>(lab))
		unblockWaitingHelping();

	/* Check for races */
	if (llvm::isa<UnlockWriteLabel>(lab))
		checkUnlockValidity(lab);
	if (llvm::isa<BInitWriteLabel>(lab))
		checkBInitValidity(lab);
	checkFinalAnnotations(lab);

#ifdef DEBUG_LUAN
	// if (wLab) {
	// Print("after handling store");
	// PrintCoherence(getGraph());
	// }


#endif

}

void GenMCDriver::handleHpProtect(std::unique_ptr<HpProtectLabel> hpLab)
{
	if (isExecutionDrivenByGraph(&*hpLab))
		return;

	addLabelToGraph(std::move(hpLab));
}

SVal GenMCDriver::handleMalloc(std::unique_ptr<MallocLabel> aLab)
{
	auto& g = getGraph();
	auto* EE = getEE();
	auto& thr = EE->getCurThr();

	if (isExecutionDrivenByGraph(&*aLab)) {
		auto* lab = llvm::dyn_cast<MallocLabel>(g.getEventLabel(aLab->getPos()));
		BUG_ON(!lab);
		return SVal(lab->getAllocAddr().get());
	}

	/* Fix and add label to the graph; return the new address */
	if (aLab->getAllocAddr() == SAddr())
		aLab->setAllocAddr(getFreshAddr(&*aLab));
	cacheEventLabel(&*aLab);
	auto* lab = llvm::dyn_cast<MallocLabel>(addLabelToGraph(std::move(aLab)));
	return SVal(lab->getAllocAddr().get());
}

void GenMCDriver::handleFree(std::unique_ptr<FreeLabel> dLab)
{
	auto& g = getGraph();
	auto* EE = getEE();
	auto& thr = EE->getCurThr();

	if (isExecutionDrivenByGraph(&*dLab))
		return;

	/* Find the size of the area deallocated */
	auto size = 0u;
	auto alloc = findAllocatingLabel(g, dLab->getFreedAddr());
	if (alloc) {
		size = alloc->getAllocSize();
	}

	/* Add a label with the appropriate store */
	dLab->setFreedSize(size);
	dLab->setAlloc(alloc);
	auto* lab = addLabelToGraph(std::move(dLab));
	alloc->setFree(llvm::dyn_cast<FreeLabel>(lab));

	/* Check whether there is any memory race */
	checkForRaces(lab);
}

void GenMCDriver::handleRCULockLKMM(std::unique_ptr<RCULockLabelLKMM> lLab)
{
	if (isExecutionDrivenByGraph(&*lLab))
		return;

	addLabelToGraph(std::move(lLab));
}

void GenMCDriver::handleRCUUnlockLKMM(std::unique_ptr<RCUUnlockLabelLKMM> uLab)
{
	if (isExecutionDrivenByGraph(&*uLab))
		return;

	addLabelToGraph(std::move(uLab));
}

void GenMCDriver::handleRCUSyncLKMM(std::unique_ptr<RCUSyncLabelLKMM> fLab)
{
	if (isExecutionDrivenByGraph(&*fLab))
		return;

	addLabelToGraph(std::move(fLab));
}

const MemAccessLabel* GenMCDriver::getPreviousVisibleAccessLabel(Event start) const
{
	auto& g = getGraph();
	std::vector<Event> finalReads;

	for (auto pos = start.prev(); pos.index > 0; --pos) {
		auto* lab = g.getEventLabel(pos);
		if (auto* rLab = llvm::dyn_cast<ReadLabel>(lab)) {
			if (getConf()->helper && g.isConfirming(rLab))
				continue;
			if (rLab->getRf()) {
				auto* wLab = llvm::dyn_cast<WriteLabel>(rLab->getRf());
				if (wLab && wLab->isLocal())
					continue;
				if (wLab && wLab->isFinal()) {
					finalReads.push_back(rLab->getPos());
					continue;
				}
				if (std::any_of(finalReads.begin(), finalReads.end(), [&](const Event& l) {
					auto* lLab = llvm::dyn_cast<ReadLabel>(g.getEventLabel(l));
					return lLab->getAddr() == rLab->getAddr() &&
						lLab->getSize() == rLab->getSize();
					}))
					continue;
			}
			return rLab;
		}
		if (auto* wLab = llvm::dyn_cast<WriteLabel>(lab))
			if (!wLab->isFinal() && !wLab->isLocal())
				return wLab;
	}
	return nullptr; /* none found */
}

void GenMCDriver::mootExecutionIfFullyBlocked(Event pos)
{
	auto& g = getGraph();

	auto* lab = getPreviousVisibleAccessLabel(pos);
	if (auto* rLab = llvm::dyn_cast_or_null<ReadLabel>(lab))
		if (!rLab->isRevisitable() || !rLab->wasAddedMax())
			moot();
	return;
}

void GenMCDriver::handleBlock(std::unique_ptr<BlockLabel> lab)
{
	if (isExecutionDrivenByGraph(&*lab))
		return;

	auto& g = getGraph();
	blockThreadTryMoot(lab->getPos(), lab->getType());
	return;
}

std::unique_ptr<VectorClock>
GenMCDriver::getReplayView() const
{
	auto& g = getGraph();
	auto v = g.getViewFromStamp(g.getMaxStamp());

	// handleBlock() is usually only called during normal execution
	// and hence not reproduced during replays.
	// We have to remove BlockLabels so that these will not lead
	// 	to the execution of extraneous instructions 

	for (auto i = 0u; i < g.getNumThreads(); i++)
		if (llvm::isa<BlockLabel>(g.getLastThreadLabel(i)))
			v->setMax(Event(i, v->getMax(i) - 1));
	return v;
}

void GenMCDriver::reportError(Event pos, VerificationError s,
	const std::string& err /* = "" */,
	const EventLabel* racyLab /* = nullptr */,
	bool shouldHalt /* = true */)
{
	auto& g = getGraph();
	auto& thr = getEE()->getCurThr();

	/* If we have already detected an error, no need to report another */
	if (isHalting())
		return;

	/* If we this is a replay (might happen if one LLVM instruction
	 * maps to many MC events), do not get into an infinite loop... */
	if (inReplay())
		return;

	/* Ignore soft errors under estimation mode.
	 * These are going to be reported later on anyway */
	if (!shouldHalt && inEstimationMode())
		return;

	auto* errLab = g.getEventLabel(pos);
	if (inRecoveryMode() && !isRecoveryValid(errLab)) {
		thr.block(BlockageType::Error);
		return;
	}

	/* If this is an invalid access, change the RF of the offending
	 * event to BOTTOM, so that we do not try to get its value.
	 * Don't bother updating the views */
	if (isInvalidAccessError(s) && llvm::isa<ReadLabel>(errLab))
		g.changeRf(errLab->getPos(), Event::getBottom());

	/* Print a basic error message and the graph.
	 * We have to save the interpreter state as replaying will
	 * destroy the current execution stack */
	auto iState = getEE()->saveState();

	getEE()->replayExecutionBefore(*getReplayView());

	llvm::raw_string_ostream out(result.message);

	out << (isHardError(s) ? "Error: " : "Warning: ") << s << "!\n";
	out << "Event " << errLab->getPos() << " ";
	if (racyLab != nullptr)
		out << "conflicts with event " << racyLab->getPos() << " ";
	out << "in graph:\n";
	printGraph(true, out);

	/* Print error trace leading up to the violating event(s) */
	if (getConf()->printErrorTrace) {
		printTraceBefore(errLab, out);
		if (racyLab != nullptr)
			printTraceBefore(racyLab, out);
	}

	/* Print the specific error message */
	if (!err.empty())
		out << err << "\n";

	/* Dump the graph into a file (DOT format) */
	if (!getConf()->dotFile.empty())
		dotPrintToFile(getConf()->dotFile, errLab, racyLab);

	getEE()->restoreState(std::move(iState));

	if (shouldHalt)
		halt(s);
}

bool GenMCDriver::tryOptimizeBarrierRevisits(const BIncFaiWriteLabel* sLab, std::vector<Event>& loads)
{
	if (getConf()->disableBAM)
		return false;

	/* If the barrier_wait() does not write the initial value, nothing to do */
	auto iVal = getBarrierInitValue(sLab->getAccess());
	if (sLab->getVal() != iVal)
		return true;

	/* Otherwise, revisit in place */
	auto& g = getGraph();
	auto bs = g.collectAllEvents([&](const EventLabel* lab) {
		auto* bLab = llvm::dyn_cast<BlockLabel>(lab);
		if (!bLab || bLab->getType() != BlockageType::Barrier)
			return false;
		auto* pLab = llvm::dyn_cast<BIncFaiWriteLabel>(
			g.getPreviousLabel(lab));
		return pLab->getAddr() == sLab->getAddr();
		});
	if (bs.size() > iVal.get() || loads.size() > 0)
		WARN_ONCE("bam-well-formed", "Execution not barrier-well-formed!\n");

	std::for_each(bs.begin(), bs.end(), [&](const Event& b) {
		auto* pLab = llvm::dyn_cast<BIncFaiWriteLabel>(g.getPreviousLabel(b));
		BUG_ON(!pLab);
		unblockThread(b);
		auto* rLab = llvm::dyn_cast<ReadLabel>(
			addLabelToGraph(BWaitReadLabel::create(b, pLab->getOrdering(), pLab->getAddr(),
				pLab->getSize(), pLab->getType(),
				pLab->getDeps())));
		g.changeRf(rLab->getPos(), sLab->getPos());
		rLab->setAddedMax(isCoMaximal(rLab->getAddr(), rLab->getRf()->getPos()));
		});
	return true;
}

bool GenMCDriver::tryOptimizeIPRs(const WriteLabel* sLab, std::vector<Event>& loads)
{
	if (!getConf()->ipr)
		return false;

	auto& g = getGraph();

	std::vector<Event> toIPR;
	loads.erase(std::remove_if(loads.begin(), loads.end(), [&](auto& l) {
		auto blocked = isAssumeBlocked(g.getReadLabel(l), sLab);
		if (blocked)
			toIPR.push_back(l);
		return blocked;
		}), loads.end());

	for (auto& l : toIPR)
		revisitInPlace(*constructBackwardRevisit(g.getReadLabel(l), sLab));

	/* We also have to filter out some regular revisits */
	auto pending = g.getPendingRMW(sLab);
	if (!pending.isInitializer()) {
		loads.erase(std::remove_if(loads.begin(), loads.end(), [&](auto& l) {
			auto* rLab = g.getReadLabel(l);
			auto* rfLab = rLab->getRf();
			return rLab->getAnnot() && // must be like that
				rfLab->getStamp() > rLab->getStamp() &&
				!getPrefixView(sLab).contains(rfLab->getPos());
			}), loads.end());
	}
	return false; /* we still have to perform the rest of the revisits */
}

bool GenMCDriver::tryOptimizeLocks(const WriteLabel* sLab, std::vector<Event>& loads)
{
	if (!llvm::isa<LockCasWriteLabel>(sLab) && !llvm::isa<UnlockWriteLabel>(sLab))
		return false;

	auto& g = getGraph();

	std::vector<Event> toIPR;
	loads.erase(std::remove_if(loads.begin(), loads.end(), [&](auto& l) {
		auto* rLab = g.getReadLabel(l);
		auto blocked = llvm::isa<LockCasReadLabel>(rLab) && llvm::isa<LockCasWriteLabel>(rLab->getRf());
		if (blocked)
			toIPR.push_back(l);
		return blocked;
		}), loads.end());

	for (auto& l : toIPR)
		revisitInPlace(*constructBackwardRevisit(g.getReadLabel(l), sLab));
	return false; /* we still have to perform the rest of the revisits */
}

void GenMCDriver::optimizeUnconfirmedRevisits(const WriteLabel* sLab, std::vector<Event>& loads)
{
	if (!getConf()->helper)
		return;

	auto& g = getGraph();

	/* If there is already a write with the same value, report a possible ABA */
	auto valid = std::count_if(store_begin(g, sLab->getAddr()), store_end(g, sLab->getAddr()),
		[&](auto& wLab) {
			return wLab.getPos() != sLab->getPos() && wLab.getVal() == sLab->getVal();
		});
	if (sLab->getAddr().isStatic() &&
		getWriteValue(g.getEventLabel(Event::getInit()), sLab->getAccess()) == sLab->getVal())
		++valid;
	WARN_ON_ONCE(valid > 0, "helper-aba-found",
		"Possible ABA pattern! Consider running without -helper.\n");

	/* Do not bother with revisits that will be unconfirmed/lead to ABAs */
	loads.erase(std::remove_if(loads.begin(), loads.end(), [&](const Event& l) {
		auto* lab = llvm::dyn_cast<ReadLabel>(g.getEventLabel(l));
		if (!g.isConfirming(lab))
			return false;

		auto sc = Event::getInit();
		auto* pLab = llvm::dyn_cast<ReadLabel>(
			g.getEventLabel(g.getMatchingSpeculativeRead(lab->getPos(), &sc)));
		ERROR_ON(!pLab, "Confirming CAS annotation error! "
			"Does a speculative read precede the confirming operation?\n");

		return sc.isInitializer();
		}), loads.end());
}

bool GenMCDriver::isConflictingNonRevBlocker(const EventLabel* pLab, const WriteLabel* sLab, const Event& s)
{
	auto& g = getGraph();
	auto* sLab2 = llvm::dyn_cast<WriteLabel>(g.getEventLabel(s));
	if (sLab2->getPos() == sLab->getPos() || !g.isRMWStore(sLab2))
		return false;
	auto& prefix = getPrefixView(sLab);
	if (prefix.contains(sLab2->getPos()) &&
		!(pLab && pLab->getStamp() < sLab2->getStamp()))
		return false;
	if (sLab2->getThread() <= sLab->getThread())
		return false;
	return std::any_of(sLab2->readers_begin(), sLab2->readers_end(), [&](auto& rLab) {
		return rLab.getStamp() < sLab2->getStamp() &&
			!prefix.contains(rLab.getPos());
		});
}

bool GenMCDriver::tryOptimizeRevBlockerAddition(const WriteLabel* sLab, std::vector<Event>& loads)
{
	if (!sLab->hasAttr(WriteAttr::RevBlocker))
		return false;

	auto& g = getGraph();
	auto* pLab = getPreviousVisibleAccessLabel(sLab->getPos().prev());
	if (std::find_if(store_begin(g, sLab->getAddr()), store_end(g, sLab->getAddr()),
		[this, pLab, sLab](auto& lab) {
			return isConflictingNonRevBlocker(pLab, sLab, lab.getPos());
		}) != store_end(g, sLab->getAddr())) {
		moot();
		loads.clear();
		return true;
	}
	return false;
}

bool GenMCDriver::tryOptimizeRevisits(const WriteLabel* sLab, std::vector<Event>& loads)
{
	auto& g = getGraph();

	/* BAM */
	if (!getConf()->disableBAM) {
		if (auto* faiLab = llvm::dyn_cast<BIncFaiWriteLabel>(sLab)) {
			if (tryOptimizeBarrierRevisits(faiLab, loads))
				return true;
		}
	}

	/* IPR + locks */
	if (getConf()->ipr) {
		if (tryOptimizeIPRs(sLab, loads))
			return true;
	}
	if (tryOptimizeLocks(sLab, loads))
		return true;

	/* Helper: 1) Do not bother with revisits that will lead to unconfirmed reads
			   2) Do not bother exploring if a RevBlocker is being re-added	*/
	if (getConf()->helper) {
		optimizeUnconfirmedRevisits(sLab, loads);
		if (sLab->hasAttr(WriteAttr::RevBlocker) && tryOptimizeRevBlockerAddition(sLab, loads))
			return true;
	}
	return false;
}

bool GenMCDriver::isAssumeBlocked(const ReadLabel* rLab, const WriteLabel* sLab)
{
	auto& g = getGraph();
	using Evaluator = SExprEvaluator<ModuleID::ID>;

	return !llvm::isa<CasReadLabel>(rLab) &&
		rLab->getAnnot() &&
		!Evaluator().evaluate(rLab->getAnnot(), getReadValue(rLab));
}

void GenMCDriver::revisitInPlace(const BackwardRevisit& br)
{
	auto& g = getGraph();
	auto* rLab = g.getReadLabel(br.getPos());
	const auto* sLab = g.getWriteLabel(br.getRev());

	BUG_ON(!llvm::isa<ReadLabel>(rLab));
	if (g.getNextLabel(rLab))
		g.removeLast(rLab->getThread());
	g.changeRf(rLab->getPos(), sLab->getPos());
	rLab->setAddedMax(true); // always true for atomicity violations
	rLab->setIPRStatus(true);

	completeRevisitedRMW(rLab);

	GENMC_DEBUG(LOG(VerbosityLevel::Debug1) << "--- In-place revisiting "
		<< rLab->getPos() << " <-- " << sLab->getPos() << "\n" << getGraph(); );

	EE->resetThread(rLab->getThread());
	EE->getThrById(rLab->getThread()).ECStack = EE->getThrById(rLab->getThread()).initEC;
	threadPrios = { rLab->getPos() };
}

void updatePredsWithPrefixView(const ExecutionGraph& g, VectorClock& preds, const VectorClock& pporf)
{
	Print("updatePredsWithPrefixView:", preds, pporf);
	/* In addition to taking (preds U pporf), make sure pporf includes rfis */
	preds.update(pporf);

	if (!dynamic_cast<const DepExecutionGraph*>(&g))
		return;
	auto& predsD = *llvm::dyn_cast<DepView>(&preds);
	for (auto i = 0u; i < pporf.size(); i++) {
		for (auto j = 1; j <= pporf.getMax(i); j++) {
			auto* lab = g.getEventLabel(Event(i, j));
			if (auto* rLab = llvm::dyn_cast<ReadLabel>(lab)) {
				if (preds.contains(rLab->getPos()) && !preds.contains(rLab->getRf())) {
					if (rLab->getRf()->getThread() == rLab->getThread())
						predsD.removeHole(rLab->getRf()->getPos());
				}
			}
			if (g.isRMWStore(lab) && pporf.contains(lab->getPos().prev()))
				predsD.removeHole(lab->getPos());
		}
	}
	return;
}

std::unique_ptr<VectorClock>
GenMCDriver::getRevisitView(const ReadLabel* rLab, const WriteLabel* sLab, const WriteLabel* midLab /* = nullptr */) const
{
	auto& g = getGraph();
	auto preds = g.getPredsView(rLab->getPos());

	updatePredsWithPrefixView(g, *preds, getPrefixView(sLab));
	if (midLab)
		updatePredsWithPrefixView(g, *preds, getPrefixView(midLab));
	return std::move(preds);
}

std::unique_ptr<BackwardRevisit>
GenMCDriver::constructBackwardRevisit(const ReadLabel* rLab, const WriteLabel* sLab) const
{
	if (!getConf()->helper)
		return std::make_unique<BackwardRevisit>(rLab, sLab, getRevisitView(rLab, sLab));

	// Print(RED("has helper in config"));
	// getchar();

	auto& g = getGraph();

	/* Check whether there is a conflicting RevBlocker */
	auto pending = g.getPendingRMW(sLab);
	auto* pLab = llvm::dyn_cast_or_null<WriteLabel>(g.getNextLabel(pending));
	pending = (!pending.isInitializer() && pLab->hasAttr(WriteAttr::RevBlocker)) ?
		pending.next() : Event::getInit();

	/* If there is, do an optimized backward revisit */
	auto& prefix = getPrefixView(sLab); // ehivh psty of g will be kept
	if (!pending.isInitializer() &&
		!getPrefixView(g.getEventLabel(pending)).contains(rLab->getPos()) &&
		rLab->getStamp() < g.getEventLabel(pending)->getStamp() &&
		!prefix.contains(pending))
		return std::make_unique<BackwardRevisitHELPER>(rLab->getPos(), sLab->getPos(),
			getRevisitView(rLab, sLab, g.getWriteLabel(pending)), pending);
	return std::make_unique<BackwardRevisit>(rLab, sLab, getRevisitView(rLab, sLab));
}

bool GenMCDriver::prefixContainsMatchingLock(const BackwardRevisit& r, const EventLabel* lab)
{
	if (!llvm::isa<UnlockWriteLabel>(lab))
		return false;
	auto l = getGraph().getMatchingLock(lab->getPos());
	if (l.isInitializer())
		return false;
	if (getPrefixView(getGraph().getWriteLabel(r.getRev())).contains(l))
		return true;
	if (auto* br = llvm::dyn_cast<BackwardRevisitHELPER>(&r))
		return getPrefixView(getGraph().getWriteLabel(br->getMid())).contains(l);
	return false;
}

bool isFixedHoleInView(const ExecutionGraph& g, const EventLabel* lab, const DepView& v)
{
	if (auto* wLabB = llvm::dyn_cast<WriteLabel>(lab))
		return std::any_of(wLabB->readers_begin(), wLabB->readers_end(),
			[&v](auto& oLab) { return v.contains(oLab.getPos()); });

	auto* rLabB = llvm::dyn_cast<ReadLabel>(lab);
	if (!rLabB)
		return false;

	/* If prefix has same address load, we must read from the same write */
	for (auto i = 0u; i < v.size(); i++) {
		for (auto j = 0u; j <= v.getMax(i); j++) {
			if (!v.contains(Event(i, j)))
				continue;
			if (auto* mLab = g.getReadLabel(Event(i, j)))
				if (mLab->getAddr() == rLabB->getAddr() && mLab->getRf() == rLabB->getRf())
					return true;
		}
	}

	if (g.isRMWLoad(rLabB)) {
		auto* wLabB = g.getWriteLabel(rLabB->getPos().next());
		return std::any_of(wLabB->readers_begin(), wLabB->readers_end(),
			[&v](auto& oLab) { return v.contains(oLab.getPos()); });
	}
	return false;
}

bool GenMCDriver::prefixContainsSameLoc(const BackwardRevisit& r,
	const EventLabel* lab) const
{
	if (!getConf()->isDepTrackingModel)
		return false;

	/* Some holes need to be treated specially. However, it is _wrong_ to keep
	 * porf views around. What we should do instead is simply check whether
	 * an event is "part" of WLAB's pporf view (even if it is not contained in it).
	 * Similar actions are taken in {WB,MO}Calculator */
	auto& g = getGraph();
	auto& v = *llvm::dyn_cast<DepView>(&getPrefixView(g.getEventLabel(r.getRev())));
	if (lab->getIndex() <= v.getMax(lab->getThread()) && isFixedHoleInView(g, lab, v))
		return true;
	if (auto* br = llvm::dyn_cast<BackwardRevisitHELPER>(&r)) {
		auto& hv = *llvm::dyn_cast<DepView>(&getPrefixView(g.getEventLabel(br->getMid())));
		return lab->getIndex() <= hv.getMax(lab->getThread()) && isFixedHoleInView(g, lab, hv);
	}
	return false;
}

bool GenMCDriver::hasBeenRevisitedByDeleted(const BackwardRevisit& r,
	const EventLabel* eLab)
{
	auto* lab = llvm::dyn_cast<ReadLabel>(eLab);
	if (!lab || lab->isIPR())
		return false;

	auto* rfLab = lab->getRf();
	auto& v = *r.getViewNoRel();
	return !v.contains(rfLab->getPos()) &&
		rfLab->getStamp() > lab->getStamp() &&
		!prefixContainsSameLoc(r, rfLab);
}

bool GenMCDriver::isCoBeforeSavedPrefix(const BackwardRevisit& r, const EventLabel* lab)
{
	auto* mLab = llvm::dyn_cast<MemAccessLabel>(lab);
	if (!mLab)
		return false;

	auto& g = getGraph();
	auto& v = r.getViewNoRel();
	auto w = llvm::isa<ReadLabel>(mLab) ? llvm::dyn_cast<ReadLabel>(mLab)->getRf()->getPos() : mLab->getPos();
	auto succIt = g.getWriteLabel(w) ? g.co_succ_begin(g.getWriteLabel(w)) : g.co_begin(mLab->getAddr());
	auto succE = g.getWriteLabel(w) ? g.co_succ_end(g.getWriteLabel(w)) : g.co_end(mLab->getAddr());
	return any_of(succIt, succE, [&](auto& sLab) {
		return v->contains(sLab.getPos()) &&
			(!getConf()->isDepTrackingModel ||
				mLab->getIndex() > getPrefixView(&sLab).getMax(mLab->getThread())) &&
			sLab.getPos() != r.getRev();
		});
}

bool GenMCDriver::coherenceSuccRemainInGraph(const BackwardRevisit& r)
{
	auto& g = getGraph();
	auto* wLab = g.getWriteLabel(r.getRev());
	if (g.isRMWStore(wLab))
		return true;

	auto succIt = g.co_succ_begin(wLab);
	auto succE = g.co_succ_end(wLab);
	if (succIt == succE)
		return true;

	return r.getViewNoRel()->contains(succIt->getPos());
}

bool wasAddedMaximally(const EventLabel* lab)
{
	if (auto* mLab = llvm::dyn_cast<MemAccessLabel>(lab))
		return mLab->wasAddedMax();
	if (auto* oLab = llvm::dyn_cast<OptionalLabel>(lab))
		return !oLab->isExpanded();
	return true;
}

bool GenMCDriver::isMaximalExtension(const BackwardRevisit& r)
{
	if (!coherenceSuccRemainInGraph(r))
		return false;

	auto& g = getGraph();
	auto& v = r.getViewNoRel();

	for (const auto& lab : labels(g)) {
		if ((lab.getPos() != r.getPos() && v->contains(lab.getPos())) ||
			prefixContainsSameLoc(r, &lab))
			continue;

		if (!wasAddedMaximally(&lab))
			return false;
		if (isCoBeforeSavedPrefix(r, &lab))
			return false;
		if (hasBeenRevisitedByDeleted(r, &lab))
			return false;
	}
	return true;
}

bool GenMCDriver::revisitModifiesGraph(const BackwardRevisit& r) const
{
	auto& g = getGraph();
	auto& v = r.getViewNoRel();
	for (auto i = 0u; i < g.getNumThreads(); i++) {
		if (v->getMax(i) + 1 != (long)g.getThreadSize(i) &&
			!g.getEventLabel(Event(i, v->getMax(i) + 1))->isTerminator())
			return true;
		if (!getConf()->isDepTrackingModel)
			continue;
		for (auto j = 0u; j < g.getThreadSize(i); j++) {
			auto* lab = g.getEventLabel(Event(i, j));
			if (!v->contains(lab->getPos()) && !llvm::isa<EmptyLabel>(lab) &&
				!lab->isTerminator())
				return true;
		}

	}
	return false;
}

std::unique_ptr<ExecutionGraph>
GenMCDriver::copyGraph(const BackwardRevisit* br, VectorClock* v) const
{
	Print("copying graph");
	auto& g = getGraph();

	/* Adjust the view that will be used for copying */
	auto& prefix = getPrefixView(g.getEventLabel(br->getRev()));
	if (auto* brh = llvm::dyn_cast<BackwardRevisitHELPER>(br)) {
		if (auto* dv = llvm::dyn_cast<DepView>(v)) {
			dv->addHole(brh->getMid());
			dv->addHole(brh->getMid().prev());
		}
		else {
			auto prev = v->getMax(brh->getMid().thread);
			v->setMax(Event(brh->getMid().thread, prev - 2));
		}
	}
	Print("g.getCopyUpTo(", *v, ")");
	auto og = g.getCopyUpTo(*v);


	/* Ensure the prefix of the write will not be revisitable */
	auto* revLab = og->getReadLabel(br->getPos());

	og->compressStampsAfter(revLab->getStamp());
	for (auto& lab : labels(*og)) {
		if (auto* rLab = llvm::dyn_cast<ReadLabel>(&lab)) {
			if (rLab && prefix.contains(rLab->getPos()))
				rLab->setRevisitStatus(false);
		}
	}
	return og;
}

GenMCDriver::ChoiceMap
GenMCDriver::createChoiceMapForCopy(const ExecutionGraph& og) const
{
	const auto& g = getGraph();
	const auto& choices = getChoiceMap();
	ChoiceMap result;

	for (auto& lab : labels(g)) {
		if (!og.containsPos(lab.getPos()) || !choices.count(lab.getStamp()))
			continue;

		auto oldStamp = lab.getStamp();
		auto newStamp = og.getEventLabel(lab.getPos())->getStamp();
		for (const auto& e : choices.at(oldStamp)) {
			if (og.containsPos(e))
				result[newStamp.get()].insert(e);
		}
	}
	return result;
}

bool GenMCDriver::checkRevBlockHELPER(const WriteLabel* sLab, const std::vector<Event>& loads)
{
	if (!getConf()->helper || !sLab->hasAttr(WriteAttr::RevBlocker))
		return true;

	auto& g = getGraph();
	if (std::any_of(loads.begin(), loads.end(), [this, &g, sLab](const Event& l) {
		auto* lLab = g.getLastThreadLabel(l.thread);
		auto* pLab = this->getPreviousVisibleAccessLabel(lLab->getPos());
		return (llvm::isa<BlockLabel>(lLab) ||
			getEE()->getThrById(lLab->getThread()).isBlocked()) &&
			pLab && pLab->getPos() == l;
		})) {
		moot();
		return false;
	}
	return true;
}

void GenMCDriver::updateStSpaceChoices(const std::vector<Event>& loads, const WriteLabel* sLab)
{
	auto& g = getGraph();
	auto& choices = getChoiceMap();
	for (const auto& l : loads) {
		const auto* rLab = g.getReadLabel(l);
		choices[rLab->getStamp()].insert(sLab->getPos());
	}
}

bool GenMCDriver::calcRevisits(const WriteLabel* sLab)
{
	auto& g = getGraph();
	auto loads = getRevisitableApproximation(sLab);

	GENMC_DEBUG(LOG(VerbosityLevel::Debug3) << "Revisitable: " << format(loads) << "\n"; );
	if (tryOptimizeRevisits(sLab, loads))
		return true;

	/* If operating in estimation mode, don't actually revisit */
	if (inEstimationMode()) {
		updateStSpaceChoices(loads, sLab);
		return checkAtomicity(sLab) && checkRevBlockHELPER(sLab, loads) && !isMoot();
	}

	GENMC_DEBUG(LOG(VerbosityLevel::Debug3) << "Revisitable (optimized): " << format(loads) << "\n"; );
	for (auto& l : loads) {
		auto* rLab = g.getReadLabel(l);
		BUG_ON(!rLab);

		auto br = constructBackwardRevisit(rLab, sLab);
		if (!isMaximalExtension(*br))
			break;

		addToWorklist(sLab->getStamp(), std::move(br));
	}

	return checkAtomicity(sLab) && checkRevBlockHELPER(sLab, loads) && !isMoot();
}

WriteLabel* GenMCDriver::completeRevisitedRMW(const ReadLabel* rLab)
{
	/* Handle non-RMW cases first */
	if (!llvm::isa<CasReadLabel>(rLab) && !llvm::isa<FaiReadLabel>(rLab))
		return nullptr;
	if (auto* casLab = llvm::dyn_cast<CasReadLabel>(rLab)) {
		if (getReadValue(rLab) != casLab->getExpected())
			return nullptr;
	}

	SVal result;
	WriteAttr wattr = WriteAttr::None;
	if (auto* faiLab = llvm::dyn_cast<FaiReadLabel>(rLab)) {
		/* Need to get the rf value within the if, as rLab might be a disk op,
		 * and we cannot get the value in that case (but it will also not be an RMW)  */
		auto rfVal = getReadValue(rLab);
		result = getEE()->executeAtomicRMWOperation(rfVal, faiLab->getOpVal(),
			faiLab->getSize(), faiLab->getOp());
		if (llvm::isa<BIncFaiReadLabel>(faiLab) && result == SVal(0))
			result = getBarrierInitValue(rLab->getAccess());
		wattr = faiLab->getAttr();
	}
	else if (auto* casLab = llvm::dyn_cast<CasReadLabel>(rLab)) {
		result = casLab->getSwapVal();
		wattr = casLab->getAttr();
	}
	else
		BUG();

	auto& g = getGraph();
	std::unique_ptr<WriteLabel> wLab = nullptr;

#define CREATE_COUNTERPART(name)					\
	case EventLabel::EL_## name ## Read:				\
		wLab = name##WriteLabel::create(rLab->getPos().next(),	\
						rLab->getOrdering(),	\
						rLab->getAddr(),	\
						rLab->getSize(),	\
						rLab->getType(),	\
						result,			\
						wattr);			\
	break;

	switch (rLab->getKind()) {
		CREATE_COUNTERPART(BIncFai);
		CREATE_COUNTERPART(NoRetFai);
		CREATE_COUNTERPART(Fai);
		CREATE_COUNTERPART(LockCas);
		CREATE_COUNTERPART(TrylockCas);
		CREATE_COUNTERPART(Cas);
		CREATE_COUNTERPART(HelpedCas);
		CREATE_COUNTERPART(ConfirmingCas);
	default:
		BUG();
	}
	BUG_ON(!wLab);
	cacheEventLabel(&*wLab);
	auto* lab = llvm::dyn_cast<WriteLabel>(addLabelToGraph(std::move(wLab)));
	BUG_ON(!rLab->getRf());
	if (auto* rfLab = llvm::dyn_cast<WriteLabel>(rLab->getRf())) {
		g.addStoreToCO(lab, ExecutionGraph::co_iterator(g.co_succ_begin(rfLab)));
	}
	else {
		g.addStoreToCO(lab, g.co_begin(lab->getAddr()));
	}
	g.addAlloc(findAllocatingLabel(g, lab->getAddr()), lab);
	return lab;
}

#ifdef REORDER_RMW
WriteLabel* GenMCDriver::completeRevisitedRMW(const ReadLabel* rLab, ExecutionGraph& g)
{

	/* Handle non-RMW cases first */
	if (!llvm::isa<CasReadLabel>(rLab) && !llvm::isa<FaiReadLabel>(rLab))
		return nullptr;
	if (auto* casLab = llvm::dyn_cast<CasReadLabel>(rLab)) {
		if (getReadValue(rLab) != casLab->getExpected()) {
			Print("getReadValue(rLab)", getReadValue(rLab), " != casLab->getExpected()", casLab->getExpected());
			return nullptr;
		}

	}

	SVal result;
	WriteAttr wattr = WriteAttr::None;
	Print("completeRevisitedRMW for", rLab->getPos());
	if (auto* faiLab = llvm::dyn_cast<FaiReadLabel>(rLab)) {
		/* Need to get the rf value within the if, as rLab might be a disk op,
		 * and we cannot get the value in that case (but it will also not be an RMW)  */
		auto rfVal = getReadValue(rLab);
		Print("read value of read: ", rfVal);
		result = getEE()->executeAtomicRMWOperation(rfVal, faiLab->getOpVal(),
			faiLab->getSize(), faiLab->getOp());
		if (llvm::isa<BIncFaiReadLabel>(faiLab) && result == SVal(0))
			result = getBarrierInitValue(rLab->getAccess());
		wattr = faiLab->getAttr();
	}
	else if (auto* casLab = llvm::dyn_cast<CasReadLabel>(rLab)) {
		result = casLab->getSwapVal();
		wattr = casLab->getAttr();
	}
	else
		BUG();

	// auto& g = getGraph();
	std::unique_ptr<WriteLabel> wLab = nullptr;

#define CREATE_COUNTERPART(name)					\
	case EventLabel::EL_## name ## Read:				\
		wLab = name##WriteLabel::create(rLab->getPos().next(),	\
						rLab->getOrdering(),	\
						rLab->getAddr(),	\
						rLab->getSize(),	\
						rLab->getType(),	\
						result,			\
						wattr);			\
		break;

	switch (rLab->getKind()) {
		CREATE_COUNTERPART(BIncFai);
		CREATE_COUNTERPART(NoRetFai);
		CREATE_COUNTERPART(Fai);
		CREATE_COUNTERPART(LockCas);
		CREATE_COUNTERPART(TrylockCas);
		CREATE_COUNTERPART(Cas);
		CREATE_COUNTERPART(HelpedCas);
		CREATE_COUNTERPART(ConfirmingCas);
	default:
		BUG();
	}
	BUG_ON(!wLab);
	// cacheEventLabel(&*wLab);
	auto* lab = llvm::dyn_cast<WriteLabel>(addLabelToGraph(std::move(wLab), g));
	Print("write label to be added:", *lab);
	// getchar();
	BUG_ON(!rLab->getRf());
	if (auto* rfLab = llvm::dyn_cast<WriteLabel>(rLab->getRf())) {

		g.addStoreToCO(lab, ExecutionGraph::co_iterator(g.co_succ_begin(rfLab)));
	}
	else {
		g.addStoreToCO(lab, g.co_begin(lab->getAddr()));
	}
	g.addAlloc(findAllocatingLabel(g, lab->getAddr()), lab);
	return lab;
}

#endif

bool GenMCDriver::revisitWrite(const WriteForwardRevisit& ri)
{
	auto& g = getGraph();
	auto* wLab = g.getWriteLabel(ri.getPos());
	BUG_ON(!wLab);

	g.removeStoreFromCO(wLab);
	g.addStoreToCO(wLab, ExecutionGraph::co_iterator(g.getWriteLabel(ri.getSucc())));
	wLab->setAddedMax(false);
	return calcRevisits(wLab);
}

bool GenMCDriver::revisitOptional(const OptionalForwardRevisit& oi)
{
	auto& g = getGraph();
	auto* oLab = llvm::dyn_cast<OptionalLabel>(g.getEventLabel(oi.getPos()));

	--result.exploredBlocked;
	BUG_ON(!oLab);
	oLab->setExpandable(false);
	oLab->setExpanded(true);
	return true;
}

bool GenMCDriver::revisitRead(const Revisit& ri)
{
	BUG_ON(!llvm::isa<ReadRevisit>(&ri));

	/* We are dealing with a read: change its reads-from and also check
	 * whether a part of an RMW should be added */
	auto& g = getGraph();
	auto* rLab = llvm::dyn_cast<ReadLabel>(g.getEventLabel(ri.getPos()));
	auto rev = llvm::dyn_cast<ReadRevisit>(&ri)->getRev();
	BUG_ON(!rLab);
#ifdef FUZZ_LUAN
	Print("revisiting read...");
	// Print("before changing rf: ", g);
	// Print("rev = ", rev);
#endif

	g.changeRf(rLab->getPos(), rev);
#ifdef FUZZ_LUAN
	// Print("after changing rf: ", g);
	// getchar();
#endif
	auto* fri = llvm::dyn_cast<ReadForwardRevisit>(&ri);
	rLab->setAddedMax(fri ? fri->isMaximal() : isCoMaximal(rLab->getAddr(), rev));
	rLab->setIPRStatus(false);

	GENMC_DEBUG(LOG(VerbosityLevel::Debug1)
		<< "--- " << (llvm::isa<BackwardRevisit>(ri) ? "Backward" : "Forward")
		<< " revisiting " << ri.getPos() << " <-- " << rev << "\n" << getGraph(); );

	/* If the revisited label became an RMW, add the store part and revisit */
	if (auto* sLab = completeRevisitedRMW(rLab))
		return calcRevisits(sLab);

	/* Blocked lock -> prioritize locking thread */
	if (llvm::isa<LockCasReadLabel>(rLab)) {
		blockThread(rLab->getPos().next(), BlockageType::LockNotAcq);
		threadPrios = { rLab->getRf()->getPos() };
	}
	auto* oLab = g.getPreviousLabelST(rLab, [&](const EventLabel* oLab) {
		return llvm::isa<SpeculativeReadLabel>(oLab);
		});

	if (llvm::isa<SpeculativeReadLabel>(rLab) || oLab)
		threadPrios = { rLab->getPos() };
	return true;
}

bool GenMCDriver::forwardRevisit(const ForwardRevisit& fr)
{
	auto& g = getGraph();
	auto* lab = g.getEventLabel(fr.getPos());
	if (auto* mi = llvm::dyn_cast<WriteForwardRevisit>(&fr))
		return revisitWrite(*mi);
	if (auto* oi = llvm::dyn_cast<OptionalForwardRevisit>(&fr))
		return revisitOptional(*oi);
	if (auto* rr = llvm::dyn_cast<RerunForwardRevisit>(&fr))
		return true;
	auto* ri = llvm::dyn_cast<ReadForwardRevisit>(&fr);
	BUG_ON(!ri);
	return revisitRead(*ri);
}

bool GenMCDriver::backwardRevisit(const BackwardRevisit& br)
{
	auto& g = getGraph();

	/* Recalculate the view because some B labels might have been
	 * removed */
	auto* brh = llvm::dyn_cast<BackwardRevisitHELPER>(&br);
	auto v = getRevisitView(g.getReadLabel(br.getPos()),
		g.getWriteLabel(br.getRev()),
		brh ? g.getWriteLabel(brh->getMid()) : nullptr);


#ifdef DEBUG_LUAN
	{
		Print("revisit view: ", *v);
		auto sLab = g.getWriteLabel(br.getRev());
		auto preds = g.getPredsView(sLab->getPos());	// cutToStamp
		Print("sLab's preds view: ", *preds);
		// if (*preds < *v) return false;
		for (int i = 0; i < g.getNumThreads(); i++) {
			if (preds->getMax(i) < v->getMax(i)) {
				return false;
			}
		}
	}


#endif

	auto og = copyGraph(&br, &*v);
	auto m = createChoiceMapForCopy(*og);
	Print(RED("back revist: copied graph og:"), *og);

	pushExecution({ std::move(og), LocalQueueT(), std::move(m) });

	repairDanglingReads(getGraph());
	auto ok = revisitRead(br);
	BUG_ON(!ok);
#ifdef FUZZ_BACKWARD

	Print(getGraph());
	Print(RED("copied from view: "), *v);
	if (auto md = brh ? g.getWriteLabel(brh->getMid()) : nullptr; md)
		Print(RED("intermediat write is:"), md->getPos());
#endif

	/* If there are idle workers in the thread pool,
	 * try submitting the job instead */
	auto* tp = getThreadPool();
	if (tp && tp->getRemainingTasks() < 8 * tp->size()) {
		if (isRevisitValid(br))
			tp->submit(extractState());
		return false;
	}
	return true;
}

bool GenMCDriver::restrictAndRevisit(Stamp stamp, const WorkSet::ItemT& item)
{
	/* First, appropriately restrict the worklist and the graph */
	Print(GREEN("restricting stamp:"), stamp);
	// getchar();
	getExecution().restrict(stamp);
	Print(GREEN("after restrict()..."));
	Print(getGraph());
	lastAdded = item->getPos();
	Print(GREEN("last added:"), lastAdded);
	if (auto* fr = llvm::dyn_cast<ForwardRevisit>(&*item)) {
		Print(RED("forward revisit"), *fr);
		return forwardRevisit(*fr);
	}

	if (auto* br = llvm::dyn_cast<BackwardRevisit>(&*item)) {
		Print(RED("backward revisit"), *br);
		return backwardRevisit(*br);
	}
	BUG();
	return false;
}

SVal GenMCDriver::handleDskRead(std::unique_ptr<DskReadLabel> drLab)
{
	auto& g = getGraph();
	auto* EE = getEE();

	if (isExecutionDrivenByGraph(&*drLab)) {
		auto* rLab = llvm::dyn_cast<DskReadLabel>(g.getEventLabel(drLab->getPos()));
		BUG_ON(!rLab);
		return getDskReadValue(rLab);
	}

	/* Make the graph aware of a (potentially) new memory location */
	g.trackCoherenceAtLoc(drLab->getAddr());

	/* Get all stores to this location from which we can read from */
	auto validStores = getRfsApproximation(&*drLab);
	BUG_ON(validStores.empty());

	/* ... and add an appropriate label with a particular rf */
	if (inRecoveryMode())
		drLab->setOrdering(llvm::AtomicOrdering::Monotonic);
	auto* lab = llvm::dyn_cast<DskReadLabel>(addLabelToGraph(std::move(drLab)));
	g.changeRf(lab->getPos(), validStores[0]);

	/* ... filter out all option that make the recovery invalid */
	filterInvalidRecRfs(lab, validStores);

	/* Push all the other alternatives choices to the Stack */
	for (auto it = validStores.begin() + 1; it != validStores.end(); ++it)
		addToWorklist(lab->getStamp(), std::make_unique<ReadForwardRevisit>(lab->getPos(), *it));
	return getDskWriteValue(g.getEventLabel(validStores[0]), lab->getAccess());
}

void GenMCDriver::handleDskWrite(std::unique_ptr<DskWriteLabel> wLab)
{
	if (isExecutionDrivenByGraph(&*wLab))
		return;

	auto& g = getGraph();

	g.trackCoherenceAtLoc(wLab->getAddr());

	/* Disk writes should always be hb-ordered */
	auto placesRange = getCoherentPlacings(wLab->getAddr(), wLab->getPos(), false);
	BUG_ON(placesRange.begin() != placesRange.end());

	/* Safe to _only_ add it at the end of MO */
	auto* lab = llvm::dyn_cast<WriteLabel>(addLabelToGraph(std::move(wLab)));
	g.addStoreToCO(lab, placesRange.end());

	calcRevisits(lab);
	return;
}

SVal GenMCDriver::handleDskOpen(std::unique_ptr<DskOpenLabel> oLab)
{
	auto& g = getGraph();

	if (isExecutionDrivenByGraph(&*oLab)) {
		auto* lab = llvm::dyn_cast<DskOpenLabel>(g.getEventLabel(oLab->getPos()));
		BUG_ON(!lab);
		return lab->getFd();
	}

	/* We get a fresh file descriptor for this open() */
	auto fd = getFreshFd();
	ERROR_ON(fd == -1, "Too many calls to open()!\n");

	oLab->setFd(SVal(fd));
	auto* lab = llvm::dyn_cast<DskOpenLabel>(addLabelToGraph(std::move(oLab)));
	return lab->getFd();
}

void GenMCDriver::handleDskFsync(std::unique_ptr<DskFsyncLabel> fLab)
{
	if (isExecutionDrivenByGraph(&*fLab))
		return;

	addLabelToGraph(std::move(fLab));
	return;
}

void GenMCDriver::handleDskSync(std::unique_ptr<DskSyncLabel> fLab)
{
	if (isExecutionDrivenByGraph(&*fLab))
		return;

	addLabelToGraph(std::move(fLab));
	return;
}

void GenMCDriver::handleDskPbarrier(std::unique_ptr<DskPbarrierLabel> fLab)
{
	if (isExecutionDrivenByGraph(&*fLab))
		return;

	addLabelToGraph(std::move(fLab));
	return;
}

bool GenMCDriver::handleHelpingCas(std::unique_ptr<HelpingCasLabel> hLab)
{
	if (isExecutionDrivenByGraph(&*hLab))
		return true;

	/* Before adding it to the graph, ensure that the helped CAS exists */
	auto& thr = getEE()->getCurThr();
	if (!checkHelpingCasCondition(&*hLab)) {
		blockThread(hLab->getPos(), BlockageType::HelpedCas);
		return false;
	}
	addLabelToGraph(std::move(hLab));
	return true;
}

bool GenMCDriver::handleOptional(std::unique_ptr<OptionalLabel> lab)
{
	auto& g = getGraph();

	if (isExecutionDrivenByGraph(&*lab))
		return llvm::dyn_cast<OptionalLabel>(g.getEventLabel(lab->getPos()))->isExpanded();

	if (std::any_of(label_begin(g), label_end(g), [&](auto& lab) {
		auto* oLab = llvm::dyn_cast<OptionalLabel>(&lab);
		return oLab && !oLab->isExpandable();
		}))
		lab->setExpandable(false);

	auto* oLab = llvm::dyn_cast<OptionalLabel>(addLabelToGraph(std::move(lab)));

	if (!inEstimationMode() && oLab->isExpandable())
		addToWorklist(oLab->getStamp(), std::make_unique<OptionalForwardRevisit>(oLab->getPos()));
	return false; /* should not be expanded yet */
}

void GenMCDriver::handleLoopBegin(std::unique_ptr<LoopBeginLabel> bLab)
{
	if (isExecutionDrivenByGraph(&*bLab))
		return;

	addLabelToGraph(std::move(bLab));
	return;
}

bool GenMCDriver::isWriteEffectful(const WriteLabel* wLab)
{
	auto& g = getGraph();
	auto* xLab = llvm::dyn_cast<FaiWriteLabel>(wLab);
	auto* rLab = llvm::dyn_cast<FaiReadLabel>(g.getPreviousLabel(wLab));
	if (!xLab || rLab->getOp() != llvm::AtomicRMWInst::BinOp::Xchg)
		return true;

	return getReadValue(rLab) != xLab->getVal();
}

bool GenMCDriver::isWriteObservable(const WriteLabel* wLab)
{
	if (wLab->isAtLeastRelease() || !wLab->getAddr().isDynamic())
		return true;

	auto& g = getGraph();
	auto* mLab = g.getPreviousLabelST(wLab, [wLab](const EventLabel* lab) {
		if (auto* aLab = llvm::dyn_cast<MallocLabel>(lab)) {
			if (aLab->contains(wLab->getAddr()))
				return true;
		}
		return false;
		});
	if (mLab == nullptr)
		return true;

	for (auto j = mLab->getIndex() + 1; j < wLab->getIndex(); j++) {
		auto* lab = g.getEventLabel(Event(wLab->getThread(), j));
		if (lab->isAtLeastRelease())
			return true;
		/* The location must not be read (loop counter) */
		if (auto* rLab = llvm::dyn_cast<ReadLabel>(lab))
			if (rLab->getAddr() == wLab->getAddr())
				return true;
	}
	return false;
}

void GenMCDriver::handleSpinStart(std::unique_ptr<SpinStartLabel> lab)
{
	auto& g = getGraph();

	/* If it has not been added to the graph, do so */
	if (isExecutionDrivenByGraph(&*lab))
		return;

	auto* stLab = addLabelToGraph(std::move(lab));

	/* Check whether we can detect some spinloop dynamically */
	auto* lbLab = g.getPreviousLabelST(stLab, [](const EventLabel* lab) {
		return llvm::isa<LoopBeginLabel>(lab);
		});
	/* If we did not find a loop-begin, this a manual instrumentation(?); report to user */
	ERROR_ON(!lbLab, "No loop-beginning found!\n");

	auto* pLab = g.getPreviousLabelST(stLab, [lbLab](const EventLabel* lab) {
		return llvm::isa<SpinStartLabel>(lab) && lab->getIndex() > lbLab->getIndex();
		});
	if (!pLab)
		return;

	for (auto i = pLab->getIndex() + 1; i < stLab->getIndex(); i++) {
		auto* wLab = llvm::dyn_cast<WriteLabel>(g.getEventLabel(Event(stLab->getThread(), i)));
		if (wLab && isWriteEffectful(wLab) && isWriteObservable(wLab))
			return; /* found event w/ side-effects */
	}
	/* Spinloop detected */
	auto stPos = stLab->getPos();
	g.removeLast(stPos.thread);
	blockThreadTryMoot(stPos, BlockageType::Spinloop);
	return;
}

bool GenMCDriver::areFaiZNEConstraintsSat(const FaiZNESpinEndLabel* lab)
{
	auto& g = getGraph();

	/* Check that there are no other side-effects since the previous iteration.
	*We don't have to look for a BEGIN label since ZNE labels are always
	*preceded by a spin - start */
	auto* ssLab = g.getPreviousLabelST(lab, [](const EventLabel* lab) {
		return llvm::isa<SpinStartLabel>(lab);
		});
	BUG_ON(!ssLab);
	for (auto i = ssLab->getIndex() + 1; i < lab->getIndex(); ++i) {
		auto* oLab = g.getEventLabel(Event(ssLab->getThread(), i));
		if (llvm::isa<WriteLabel>(oLab) && !llvm::isa<FaiWriteLabel>(oLab))
			return false;
	}

	auto* wLab = llvm::dyn_cast<FaiWriteLabel>(
		g.getPreviousLabelST(lab, [](const EventLabel* lab) { return llvm::isa<FaiWriteLabel>(lab); }));
	BUG_ON(!wLab);

	/* All stores in the RMW chain need to be read from at most 1 read,
	 * and there need to be no other stores that are not hb-before lab */
	for (auto& lab : labels(g)) {
		if (auto* mLab = llvm::dyn_cast<MemAccessLabel>(&lab)) {
			if (mLab->getAddr() == wLab->getAddr() && !llvm::isa<FaiReadLabel>(mLab) &&
				!llvm::isa<FaiWriteLabel>(mLab) && !getHbView(wLab).contains(mLab->getPos()))
				return false;
		}
	}
	return true;
}

void GenMCDriver::handleFaiZNESpinEnd(std::unique_ptr<FaiZNESpinEndLabel> lab)
{
	auto& g = getGraph();
	auto* EE = getEE();

	/* If we are actually replaying this one, it is not a spin loop*/
	if (isExecutionDrivenByGraph(&*lab))
		return;

	auto* zLab = llvm::dyn_cast<FaiZNESpinEndLabel>(addLabelToGraph(std::move(lab)));
	if (areFaiZNEConstraintsSat(&*zLab)) {
		auto pos = zLab->getPos();
		g.removeLast(pos.thread);
		blockThreadTryMoot(pos, BlockageType::FaiZNESpinloop);
	}
	return;
}

void GenMCDriver::handleLockZNESpinEnd(std::unique_ptr<LockZNESpinEndLabel> lab)
{
	if (isExecutionDrivenByGraph(&*lab))
		return;

	blockThreadTryMoot(lab->getPos(), BlockageType::LockZNESpinloop);
	return;
}


/************************************************************
 ** Printing facilities
 ***********************************************************/

static void executeMDPrint(const EventLabel* lab,
	const std::pair<int, std::string>& locAndFile,
	std::string inputFile,
	llvm::raw_ostream& os = llvm::outs())
{
	std::string errPath = locAndFile.second;
	Parser::stripSlashes(errPath);
	Parser::stripSlashes(inputFile);

	os << " ";
	if (errPath != inputFile)
		os << errPath << ":";
	else
		os << "L.";
	os << locAndFile.first;
}

/* Returns true if the corresponding LOC should be printed for this label type */
bool shouldPrintLOC(const EventLabel* lab)
{
	/* Begin/End labels don't have a corresponding LOC */
	if (llvm::isa<ThreadStartLabel>(lab) ||
		llvm::isa<ThreadFinishLabel>(lab))
		return false;

	/* Similarly for allocations that don't come from malloc() */
	if (auto* mLab = llvm::dyn_cast<MallocLabel>(lab))
		return mLab->getAllocAddr().isHeap() && !mLab->getAllocAddr().isInternal();

	return true;
}

std::string GenMCDriver::getVarName(const SAddr& addr) const
{
	if (addr.isStatic())
		return getEE()->getStaticName(addr);

	const auto& g = getGraph();
	auto a = g.getMalloc(addr);

	if (a.isInitializer())
		return "???";

	auto* aLab = llvm::dyn_cast<MallocLabel>(g.getEventLabel(a));
	BUG_ON(!aLab);
	if (aLab->getNameInfo())
		return aLab->getName() +
		aLab->getNameInfo()->getNameAtOffset(addr - aLab->getAllocAddr());
	return "";
}

#ifdef ENABLE_GENMC_DEBUG
llvm::raw_ostream::Colors getLabelColor(const EventLabel* lab)
{
	auto* mLab = llvm::dyn_cast<MemAccessLabel>(lab);
	if (!mLab)
		return llvm::raw_ostream::Colors::WHITE;

	if (llvm::isa<ReadLabel>(mLab) && !llvm::dyn_cast<ReadLabel>(mLab)->isRevisitable())
		return llvm::raw_ostream::Colors::RED;
	if (mLab->wasAddedMax())
		return llvm::raw_ostream::Colors::GREEN;
	return llvm::raw_ostream::Colors::WHITE;
}
#endif

void GenMCDriver::printGraph(bool printMetadata /* false */, llvm::raw_ostream& s /* = llvm::dbgs() */)
{
	auto& g = getGraph();
	LabelPrinter printer([this](const SAddr& saddr) { return getVarName(saddr); },
		[this](const ReadLabel& lab) {
			return llvm::isa<DskReadLabel>(&lab) ?
				getDskReadValue(llvm::dyn_cast<DskReadLabel>(&lab)) :
				getReadValue(&lab);
		});

	/* Print the graph */
	for (auto i = 0u; i < g.getNumThreads(); i++) {
		auto& thr = EE->getThrById(i);
		s << thr;
		if (getConf()->symmetryReduction) {
			if (auto* bLab = g.getFirstThreadLabel(i)) {
				auto symm = bLab->getSymmetricTid();
				if (symm != -1) s << " symmetric with " << symm;
			}
		}
		s << ":\n";
		for (auto j = 1u; j < g.getThreadSize(i); j++) {
			auto* lab = g.getEventLabel(Event(i, j));
			s << "\t";
			GENMC_DEBUG(
				if (getConf()->colorAccesses)
					s.changeColor(getLabelColor(lab));
			);
			s << printer.toString(*lab);
			GENMC_DEBUG(s.resetColor(););
			GENMC_DEBUG(if (getConf()->printStamps) s << " @ " << lab->getStamp(); );
			if (printMetadata && thr.prefixLOC[j].first && shouldPrintLOC(lab)) {
				executeMDPrint(lab, thr.prefixLOC[j], getConf()->inputFile, s);
			}
			s << "\n";
		}
	}

	/* MO: Print coherence information */
	auto header = false;
	for (auto locIt = g.loc_begin(), locE = g.loc_end(); locIt != locE; ++locIt) {
		/* Skip empty and single-store locations */
		if (g.hasLocMoreThanOneStore(locIt->first)) {
			if (!header) {
				s << "Coherence:\n";
				header = true;
			}
			auto* wLab = &*g.co_begin(locIt->first);
			s << getVarName(wLab->getAddr()) << ": [ ";
			for (const auto& w : stores(g, locIt->first))
				s << w << " ";
			s << "]\n";
		}
	}
	s << "\n";
}

void GenMCDriver::dotPrintToFile(const std::string& filename,
	const EventLabel* errLab, const EventLabel* confLab)
{
	auto& g = getGraph();
	auto* EE = getEE();
	std::ofstream fout(filename);
	llvm::raw_os_ostream ss(fout);
	DotPrinter printer([this](const SAddr& saddr) { return getVarName(saddr); },
		[this](const ReadLabel& lab) {
			return llvm::isa<DskReadLabel>(&lab) ?
				getDskReadValue(llvm::dyn_cast<DskReadLabel>(&lab)) :
				getReadValue(&lab);
		});

	auto before = getPrefixView(errLab).clone();
	if (confLab)
		before->update(getPrefixView(confLab));

	/* Create a directed graph graph */
	ss << "strict digraph {\n";
	/* Specify node shape */
	ss << "node [shape=plaintext]\n";
	/* Left-justify labels for clusters */
	ss << "labeljust=l\n";
	/* Draw straight lines */
	ss << "splines=false\n";

	/* Print all nodes with each thread represented by a cluster */
	for (auto i = 0u; i < before->size(); i++) {
		auto& thr = EE->getThrById(i);
		ss << "subgraph cluster_" << thr.id << "{\n";
		ss << "\tlabel=\"" << thr.threadFun->getName().str() << "()\"\n";
		for (auto j = 1; j <= before->getMax(i); j++) {
			auto* lab = g.getEventLabel(Event(i, j));

			ss << "\t\"" << lab->getPos() << "\" [label=<";

			/* First, print the graph label for this node */
			ss << printer.toString(*lab);

			/* And then, print the corresponding line number */
			if (thr.prefixLOC[j].first && shouldPrintLOC(lab)) {
				ss << " <FONT COLOR=\"gray\">";
				executeMDPrint(lab, thr.prefixLOC[j], getConf()->inputFile, ss);
				ss << "</FONT>";
			}

			ss << ">"
				<< (lab->getPos() == errLab->getPos() || lab->getPos() == confLab->getPos() ?
					",style=filled,fillcolor=yellow" : "")
				<< "]\n";
		}
		ss << "}\n";
	}

	/* Print relations between events (po U rf) */
	for (auto i = 0u; i < before->size(); i++) {
		auto& thr = EE->getThrById(i);
		for (auto j = 0; j <= before->getMax(i); j++) {
			auto* lab = g.getEventLabel(Event(i, j));

			/* Print a po-edge, but skip dummy start events for
			 * all threads except for the first one */
			if (j < before->getMax(i) && !llvm::isa<ThreadStartLabel>(lab))
				ss << "\"" << lab->getPos() << "\" -> \""
				<< lab->getPos().next() << "\"\n";
			if (auto* rLab = llvm::dyn_cast<ReadLabel>(lab)) {
				/* Do not print RFs from INIT, BOTTOM, and same thread */
				if (llvm::dyn_cast_or_null<WriteLabel>(rLab) &&
					rLab->getRf()->getThread() != lab->getThread()) {
					ss << "\"" << rLab->getRf() << "\" -> \""
						<< rLab->getPos() << "\"[color=green, constraint=false]\n";
				}
			}
			if (auto* bLab = llvm::dyn_cast<ThreadStartLabel>(lab)) {
				if (thr.id == 0)
					continue;
				ss << "\"" << bLab->getParentCreate() << "\" -> \""
					<< bLab->getPos().next() << "\"[color=blue, constraint=false]\n";
			}
			if (auto* jLab = llvm::dyn_cast<ThreadJoinLabel>(lab))
				ss << "\"" << g.getLastThreadEvent(jLab->getChildId()) << "\" -> \""
				<< jLab->getPos() << "\"[color=blue, constraint=false]\n";
		}
	}

	ss << "}\n";
}

void GenMCDriver::recPrintTraceBefore(const Event& e, View& a,
	llvm::raw_ostream& ss /* llvm::outs() */)
{
	const auto& g = getGraph();

	if (a.contains(e))
		return;

	auto ai = a.getMax(e.thread);
	a.setMax(e);
	auto& thr = getEE()->getThrById(e.thread);
	for (int i = ai; i <= e.index; i++) {
		const EventLabel* lab = g.getEventLabel(Event(e.thread, i));
		if (auto* rLab = llvm::dyn_cast<ReadLabel>(lab))
			if (rLab->getRf())
				recPrintTraceBefore(rLab->getRf()->getPos(), a, ss);
		if (auto* jLab = llvm::dyn_cast<ThreadJoinLabel>(lab))
			recPrintTraceBefore(g.getLastThreadEvent(jLab->getChildId()), a, ss);
		if (auto* bLab = llvm::dyn_cast<ThreadStartLabel>(lab))
			if (!bLab->getParentCreate().isInitializer())
				recPrintTraceBefore(bLab->getParentCreate(), a, ss);

		/* Do not print the line if it is an RMW write, since it will be
		 * the same as the previous one */
		if (llvm::isa<CasWriteLabel>(lab) || llvm::isa<FaiWriteLabel>(lab))
			continue;
		/* Similarly for a Wna just after the creation of a thread
		 * (it is the store of the PID) */
		if (i > 0 && llvm::isa<ThreadCreateLabel>(g.getPreviousLabel(lab)))
			continue;
		Parser::parseInstFromMData(thr.prefixLOC[i], thr.threadFun->getName().str(), ss);
	}
	return;
}

void GenMCDriver::printTraceBefore(const EventLabel* lab, llvm::raw_ostream& s /* = llvm::dbgs() */)
{
	s << "Trace to " << lab->getPos() << ":\n";

	/* Linearize (po U rf) and print trace */
	View a;
	recPrintTraceBefore(lab->getPos(), a, s);
}
