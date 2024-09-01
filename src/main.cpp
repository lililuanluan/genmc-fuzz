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
#include "DriverFactory.hpp"
#include "Error.hpp"
#include "LLVMModule.hpp"

 // luan
#include "debug_luan.hpp"
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#include <cstring>
#include <cxxabi.h>


#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <memory_resource>

auto getOutFilename(const std::shared_ptr<const Config>& /*conf*/) -> std::string
{
	return "/tmp/__genmc.ll";
}

auto buildCompilationArgs(const std::shared_ptr<const Config>& conf) -> std::string
{
	std::string args;

	args += " -fno-discard-value-names";
#ifdef HAVE_CLANG_DISABLE_OPTNONE
	args += " -Xclang";
	args += " -disable-O0-optnone";
#endif
	args += " -g"; /* Compile with -g to get debugging mdata */
	for (const auto& f : conf->cflags)
		args += " " + f;
	args += " -I" SRC_INCLUDE_DIR;
	args += " -I" INCLUDE_DIR;
	auto inodeFlag = " -D__CONFIG_GENMC_INODE_DATA_SIZE=" + std::to_string(conf->maxFileSize);
	args += " " + inodeFlag;
	args += " -S -emit-llvm";
	args += " -o " + getOutFilename(conf);
	args += " " + conf->inputFile;

	return args;
}

auto compileInput(const std::shared_ptr<const Config>& conf,
	const std::unique_ptr<llvm::LLVMContext>& ctx,
	std::unique_ptr<llvm::Module>& module) -> bool
{
	const auto* path = CLANGPATH;
	auto command = path + buildCompilationArgs(conf);
	if (std::system(command.c_str()) != 0)
		return false;

	module = LLVMModule::parseLLVMModule(getOutFilename(conf), ctx);
	return true;
}

void transformInput(const std::shared_ptr<Config>& conf,
	llvm::Module& module, ModuleInfo& modInfo)
{
	LLVMModule::transformLLVMModule(module, modInfo, conf);
	if (!conf->transformFile.empty())
		LLVMModule::printLLVMModule(module, conf->transformFile);

	/* Perhaps override the MM under which verification will take place */
	if (conf->mmDetector && modInfo.determinedMM.has_value() && isStrongerThan(*modInfo.determinedMM, conf->model)) {
		conf->model = *modInfo.determinedMM;
		conf->isDepTrackingModel = (conf->model == ModelType::IMM);
		LOG(VerbosityLevel::Tip) << "Automatically adjusting memory model to " << conf->model
			<< ". You can disable this behavior with -disable-mm-detector.\n";
	}
}

auto getElapsedSecs(const std::chrono::high_resolution_clock::time_point& begin) -> long double
{
	static constexpr long double secToMillFactor = 1e-3L;
	auto now = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count() * secToMillFactor;
}

void printEstimationResults(const std::shared_ptr<const Config>& conf,
	const std::chrono::high_resolution_clock::time_point& begin,
	const GenMCDriver::Result& res)
{
	llvm::outs() << res.message;
	llvm::outs() << (res.status == VerificationError::VE_OK ? "*** Estimation complete.\n" : "*** Estimation unsuccessful.\n");

	auto mean = std::llround(res.estimationMean);
	auto sd = std::llround(std::sqrt(res.estimationVariance));
	auto elapsed = getElapsedSecs(begin);
	auto meanTimeSecs = elapsed / (res.explored + res.exploredBlocked);
#ifdef DEBUG_LUAN
	llvm::outs() << "Estimatioin time elapsed: " << llvm::format("%.3Lf", elapsed) << "s\n";

#endif
	llvm::outs() << "Total executions estimate: " << mean << " (+- " << sd << ")\n";
	llvm::outs() << "Time to completion estimate: " << llvm::format("%.2Lf", meanTimeSecs * mean) << "s\n";
	GENMC_DEBUG(
		if (conf->printEstimationStats)
			llvm::outs() << "Estimation moot: " << res.exploredMoot << "\n"
			<< "Estimation blocked: " << res.exploredBlocked << "\n"
			<< "Estimation complete: " << res.explored << "\n";
	);
}

void printVerificationResults(const std::shared_ptr<const Config>& conf,
	const std::chrono::high_resolution_clock::time_point& begin,
	const GenMCDriver::Result& res)
{
	llvm::outs() << res.message;
	llvm::outs() << (res.status == VerificationError::VE_OK ?
		"*** Verification complete. No errors were detected.\n" : "*** Verification unsuccessful.\n");

	llvm::outs() << "Number of complete executions explored: " << res.explored << "\n";
	llvm::outs() << "Number of executions explored: " << res.explored + res.exploredBlocked;
	GENMC_DEBUG(
		llvm::outs() << ((conf->countDuplicateExecs) ?
			" (" + std::to_string(res.duplicates) + " duplicates)" : "");
	);
	if (res.exploredBlocked != 0U) {
		llvm::outs() << "\nNumber of blocked executions seen: " << res.exploredBlocked;
	}
	GENMC_DEBUG(
		if (conf->countMootExecs) {
			llvm::outs() << " (+ " << res.exploredMoot << " mooted)";
		};
	);
	llvm::outs() << "\nTotal wall-clock time: "
		<< llvm::format("%.2Lf", getElapsedSecs(begin))
		<< "s\n";
}

#ifdef CATCHSIG_LUAN
void printStackTrace() {
	const int max_frames = 128;
	void* addrlist[max_frames + 1];

	// Get the addresses of the call stack
	int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

	if (addrlen == 0) {
		std::cerr << "  <no stack trace available>\n";
		return;
	}

	// Create the symbol table
	char** symbollist = backtrace_symbols(addrlist, addrlen);

	for (int i = 1; i < addrlen; i++) {
		char* demangled = nullptr;
		char* mangled_name = nullptr;
		char* offset_begin = nullptr;
		char* offset_end = nullptr;

		// Find parentheses and +address offset surrounding mangled name
		for (char* p = symbollist[i]; *p; ++p) {
			if (*p == '(') {
				mangled_name = p;
			}
			else if (*p == '+') {
				offset_begin = p;
			}
			else if (*p == ')' && offset_begin) {
				offset_end = p;
				break;
			}
		}

		if (mangled_name && offset_begin && offset_end && mangled_name < offset_begin) {
			*mangled_name++ = '\0';
			*offset_begin++ = '\0';
			*offset_end = '\0';

			int status;
			demangled = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);

			// if (status == 0) {

			// 	std::cerr << symbollist[i] << ": " << demangled << "+" << offset_begin << std::endl;
			// }
			// else {
			// 	std::cerr << "still mangled name\n";
			// 	std::cerr << symbollist[i] << ": " << mangled_name << "+" << offset_begin << std::endl;
			// }
			free(demangled);
			std::cerr << symbollist[i] << ": " << mangled_name << "+" << offset_begin << std::endl;
			char cmd[512];
			snprintf(cmd, sizeof(cmd), "addr2line -e %s %s", symbollist[i], offset_begin);
			// std::cerr << "Command: " << cmd << std::endl;

			system(cmd);
		}
		else {
			// Just print the whole line if it doesn't match the expected format
			std::cerr << symbollist[i] << std::endl;
		}
	}
	free(symbollist);
}
void printStackTrace1() {
	const int max_frames = 128;
	void* addrlist[max_frames + 1];

	// 获取调用栈地址
	int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

	if (addrlen == 0) {
		std::cerr << "  <no stack trace available>\n";
		return;
	}

	// 创建符号表
	char** symbollist = backtrace_symbols(addrlist, addrlen);

	for (int i = 1; i < addrlen; i++) {
		char cmd[256];
		snprintf(cmd, sizeof(cmd), "addr2line -e %s %p", "../genmc", addrlist[i]);
		std::cerr << symbollist[i] << std::endl;
	}
	free(symbollist);
}

void signalHandler(int sig) {
	const char* message = "Error: signal ";
	write(STDERR_FILENO, message, strlen(message));

	char sigStr[10];
	snprintf(sigStr, sizeof(sigStr), "%d", sig);
	write(STDERR_FILENO, sigStr, strlen(sigStr));

	const char* newline = ":\n";
	write(STDERR_FILENO, newline, strlen(newline));

	printStackTrace();
	_exit(1);
}
#endif

#ifdef DEBUG_LUAN

auto main_dbg(int argc, char** argv) -> int {
#ifdef CATCHSIG_LUAN
	std::signal(SIGSEGV, signalHandler);
	std::signal(SIGABRT, signalHandler);
#endif
	std::pmr::set_default_resource(&mem_resource);


	auto begin = std::chrono::high_resolution_clock::now();
	auto conf = std::make_shared<Config>();

	conf->getConfigOptions(argc, argv);

	auto ctx = std::make_unique<llvm::LLVMContext>(); // *dtor after module's*
	std::unique_ptr<llvm::Module> module;
	if (conf->inputFromBitcodeFile) {
		module = LLVMModule::parseLLVMModule(conf->inputFile, ctx);
	}
	else if (!compileInput(conf, ctx, module)) {
		return ECOMPILE;
	}
	llvm::outs() << "*** Compilation complete.\n";

	/* Perform the necessary transformations */
	auto modInfo = std::make_unique<ModuleInfo>(*module);
	transformInput(conf, *module, *modInfo);
	llvm::outs() << "*** Transformation complete.\n";

	/* Estimate the state space */
	if (conf->estimate) {
		LOG(VerbosityLevel::Tip) << "Estimating state-space size. For better performance, you can use --disable-estimation.\n";
		// getchar();
		auto res = GenMCDriver::estimate(conf, module, modInfo);
		printEstimationResults(conf, begin, res);
		if (res.status != VerificationError::VE_OK)
			return EVERIFY;
	}
	// auto res = GenMCDriver::verify(conf, std::move(module), std::move(modInfo));
	// printVerificationResults(conf, begin, res);

	/* TODO: Check globalContext.destroy() and llvm::shutdown() */
	// return res.status == VerificationError::VE_OK ? 0 : EVERIFY;
	return 0;
}
#endif

auto main(int argc, char** argv) -> int
{
#ifdef DEBUG_LUAN
	return main_dbg(argc, argv);
#else
	auto begin = std::chrono::high_resolution_clock::now();
	auto conf = std::make_shared<Config>();

	conf->getConfigOptions(argc, argv);

	llvm::outs() << PACKAGE_NAME " v" PACKAGE_VERSION
		<< " (LLVM " LLVM_VERSION ")\n"
		<< "Copyright (C) 2023 MPI-SWS. All rights reserved.\n\n";

	auto ctx = std::make_unique<llvm::LLVMContext>(); // *dtor after module's*
	std::unique_ptr<llvm::Module> module;
	if (conf->inputFromBitcodeFile) {
		module = LLVMModule::parseLLVMModule(conf->inputFile, ctx);
	}
	else if (!compileInput(conf, ctx, module)) {
		return ECOMPILE;
	}
	llvm::outs() << "*** Compilation complete.\n";

	/* Perform the necessary transformations */
	auto modInfo = std::make_unique<ModuleInfo>(*module);
	transformInput(conf, *module, *modInfo);
	llvm::outs() << "*** Transformation complete.\n";

	/* Estimate the state space */
	if (conf->estimate) {
		LOG(VerbosityLevel::Tip) << "Estimating state-space size. For better performance, you can use --disable-estimation.\n";
		auto res = GenMCDriver::estimate(conf, module, modInfo);
		printEstimationResults(conf, begin, res);
		if (res.status != VerificationError::VE_OK)
			return EVERIFY;
	}

	/* Go ahead and try to verify */
	auto res = GenMCDriver::verify(conf, std::move(module), std::move(modInfo));
	printVerificationResults(conf, begin, res);

	/* TODO: Check globalContext.destroy() and llvm::shutdown() */
	return res.status == VerificationError::VE_OK ? 0 : EVERIFY;
#endif
}
