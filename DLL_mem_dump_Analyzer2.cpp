// ConsoleApplication3.cpp : Defines the entry point for the console application.
//
// [CRITICAL ISSUE #2 - COMPLETED] Phase 4-5 Integration Guide:
// =====================================================================
// Paper Section 3.6.7 "Phase 4-5 Integration Effect" Implementation Status:
//
// 1. Pattern Query Interface (Implemented):
//    - GetPatternFromChangeMap(map, address): Query pattern info from ChangeMap
//    - Returns change_frequency, entropy, unique_values via ChangePattern struct
//
// 2. LogMemoryChange Integration (Implemented):
//    - Function signature: void LogMemoryChange(..., MemoryChangeMap* changeMap)
//    - If changeMap is provided, query pattern via GetPatternFromChangeMap
//    - Pass pattern to CalculateClassificationScore_Phase5
//
// 3. Call Flow:
//    Phase 3 (Differential Comparison):
//    - LogMemoryChange(handle, log, &change, region, NULL) → Signal-only based classification
//
//    Phase 4 (Change Tracking):
//    - Build MemoryChangeMap (store snapshots via AddAddressHistory)
//    - Analyze change patterns via GenerateChangeLog
//
//    Integration Phase (Future):
//    - After Phase 4 completes, with changeMap in valid state
//    - LogMemoryChange(handle, log, &change, region, changeMap) call
//    - Through this, pattern is passed with actual values, improving classification accuracy
//
// 4. Pattern-based Score Adjustment:
//    within CalculateClassificationScore_Phase5:
//    - If pattern is not NULL, utilize pattern->change_frequency
//    - Utilize pattern->entropy for entropy threshold comparison
//    - Static patterns (change_frequency=0) are included if key candidate score >= 75%
//
// [Implementation Status] Structure complete, call paths configured, backward compatibility maintained
// [Testing Required] Validate actual pattern data in Phase 4 integration scenarios
//
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_STDIO_ARBITRARY_WIDE_SPECIFIERS 1

#if defined(_ARM64_) || defined(_M_ARM64)
#define MACHINE_TYPE IMAGE_FILE_MACHINE_ARM64
// ARM64 registers are defined differently - use when needed
//#define IP_REGISTER Pc
//#define FRAME_REGISTER Fp
//#define STACK_REGISTER Sp
#elif defined(_WIN64)
#define MACHINE_TYPE IMAGE_FILE_MACHINE_AMD64
#define IP_REGISTER Rip
#define FRAME_REGISTER Rbp
#define STACK_REGISTER Rsp
#else
#define MACHINE_TYPE IMAGE_FILE_MACHINE_I386
#define IP_REGISTER Eip
#define FRAME_REGISTER Ebp
#define STACK_REGISTER Esp
#endif

// Windows headers
#include <Windows.h>
#include <TlHelp32.h>
#include <heapapi.h>
#include <psapi.h>
#include <dbghelp.h>
#include <WinBase.h>
#include <tchar.h>

// Standard C/C++ headers
#include <stdio.h>
#include <string.h>
#include <cstring>
#include <cwctype>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <ctype.h>
#include <locale.h>
#include <conio.h>
#include <math.h>
#include <wchar.h>

#include <Shlwapi.h> // Required for PathFindFileName function

#pragma comment(lib, "Shlwapi.lib") // Link Shlwapi.lib library
#pragma comment(lib, "dbghelp.lib")


// Struct for storing DLL information // 23.06.17 By KJ
#define MAX_DLL_NAME_LENGTH 256
#define MAX_API_NAME_LENGTH 256
// MAX_PATH is already defined in Windows.h, so do not redefine
#ifndef MAX_PATH
#define MAX_PATH 260
#endif
#define MAX_NAME_LENGTH 260
#define BUFFER_SIZE 4096
#define MAX_SYM_NAME 256


// ============================================================
// Struct Definitions - Module and DLL Information
// ============================================================

// Struct for storing module information
typedef struct _ModuleInfo
{
	TCHAR szModName[MAX_PATH];  // Module name (full path)
	DWORD dwBaseSize;           // Base size of the module
	DWORD dwCurrentSize;        // Current size of the module in memory
} ModuleInfo;


// Struct for storing detailed DLL information // 23.06.17 By KJ
typedef struct {
	HMODULE baseAddress;  // Memory base address where DLL is loaded
	DWORD size;           // Size of the DLL (bytes)
	char apiList[MAX_API_NAME_LENGTH][MAX_DLL_NAME_LENGTH]; // 2D string array for storing external API list
	int apiCount;         // Number of loaded external APIs
} DllInfo;


// Simple module info struct (stores only process ID and base address)
typedef struct _ModuleInfo2 {
	DWORD processId;     // Process ID where the module is loaded
	LPVOID moduleBase;   // Memory base address of the module
} ModuleInfo2;


// ============================================================
// Struct Definitions - Phase 3: Memory Change Tracking
// ============================================================

// Phase 3: Struct for storing memory change information between two snapshots
typedef struct _MemoryChange {
	size_t startAddress;        // Memory address where the change started
	size_t currentAddress;      // Currently processing memory address
	unsigned char* oldData;     // Data buffer before change (previous snapshot)
	unsigned char* newData;     // Data buffer after change (current snapshot)
	size_t capacity;            // Maximum capacity of the buffer (dynamically expandable)
	size_t length;              // Actual length of the changed data (bytes)
} MemoryChange;

// Phase 2: Memory region header struct (stored in snapshot file)
// Defines snapshot file structure by storing metadata for each memory region
typedef struct _MemoryRegionHeader {
	SIZE_T baseAddress;          // Start address of the memory region
	SIZE_T regionSize;           // Total size of the region (bytes)
	char regionType[32];         // Region type ("DLL Data Section", "Private (Stack/Heap)", etc.)
	DWORD protection;            // Memory protection flags (PAGE_READWRITE, etc.) - 4 bytes
	DWORD threadId;              // Thread ID associated with this region - 4 bytes
	unsigned long timestamp;     // Snapshot creation time (Unix timestamp) - 8 bytes
} MemoryRegionHeader;

// ============================================================
// Struct Definitions - Phase 4: Per-Address Change Pattern Analysis
// ============================================================

// Snapshot-related configuration constants
#define DEFAULT_SNAPSHOTS 20            // Default snapshot count (user-configurable)
#define MAX_SNAPSHOTS_CAPACITY 256      // Maximum supported snapshot count (optimized for dynamic allocation)

// Phase 4: Struct for tracking change history of individual memory addresses
// Records value changes at a specific address across multiple snapshots in chronological order
typedef struct _AddressChangeHistory {
	SIZE_T address;                    // Memory address being tracked
	unsigned char* values;             // Dynamically allocated array: stores value at each snapshot (values[i] = value at i-th snapshot)
	int snapshot_count;                // Number of snapshots recorded so far
	int max_snapshot_capacity;         // Current allocated capacity of values array (dynamically expandable)
	double change_frequency;           // Change frequency: ratio of value changes (0.0 = static, 1.0 = changed every time)
	int total_changes;                 // Total change count: number of times value changed from previous snapshot
	char region_type[32];              // Memory region type ("DLL Data Section", "Private (Stack/Heap)", etc.)
} AddressChangeHistory;

// Phase 4: Complete memory change map (unified management of all address histories)
// Manages change histories of all tracked memory addresses in a single map
typedef struct _MemoryChangeMap {
	AddressChangeHistory* histories;  // Dynamically allocated history array (each element is one address history)
	int history_count;                // Number of unique addresses currently being tracked
	int max_histories;                // Maximum capacity of the histories array
} MemoryChangeMap;

// Phase 4: Change pattern analysis result struct
// Classifies patterns by analyzing change history of a specific address (static/dynamic/frequent changes, etc.)
typedef struct _ChangePattern {
	char pattern_name[64];         // Pattern name: "Static", "Frequently Changing", "Always Changing", etc.
	double entropy;                // Value entropy: degree of randomness (higher value indicates higher likelihood of encryption key)
	int unique_values;             // Unique value count: number of distinct values observed across snapshots
	double stability_score;        // Stability score: 0 (very unstable) ~ 100 (very stable)
	double change_frequency;       // Change frequency: 0.0 (no change) ~ 1.0 (changed every time)
} ChangePattern;

// ============================================================
// Struct Definitions - Phase 5: ML-based Multi-class Classification (Paper Section 3)
// ============================================================

// Phase 5: Cryptographic data classification enum (5 classes from the paper)
// Paper Equation (1): C = {KEY, IV, CIPHERTEXT, PLAINTEXT, NON_CRYPTO}
typedef enum _CryptoClassification {
	CRYPTO_CLASS_KEY = 0,           // Encryption key (AES, RSA, etc.)
	CRYPTO_CLASS_IV = 1,            // Initialization Vector (IV, Nonce)
	CRYPTO_CLASS_CIPHERTEXT = 2,    // Ciphertext (encrypted data)
	CRYPTO_CLASS_PLAINTEXT = 3,     // Plaintext (unencrypted data)
	CRYPTO_CLASS_NON_CRYPTO = 4     // Non-cryptographic data (general memory)
} CryptoClassification;

// Phase 5: Temporal change pattern types (Paper Signal 5: F8)
typedef enum _TemporalPatternType {
	PATTERN_STATIC = 0,             // 0-1 changes: static data (master key candidate)
	PATTERN_PARTIAL = 1,            // 2-4 changes: partial change
	PATTERN_FREQUENT = 2,           // 5-7 changes: frequent change (dynamic key)
	PATTERN_ALWAYS_CHANGING = 3     // 8+ changes: always changing (session key)
} TemporalPatternType;

// Phase 5: Memory region types (Paper Signal 4: F6)
typedef enum _MemoryRegionType {
	REGION_UNKNOWN = 0,             // Unknown region
	REGION_DLL_DATA = 1,            // DLL data section (.data, .rdata)
	REGION_STACK_HEAP = 2,          // Stack or heap allocation
	REGION_OTHER = 3                // Other region
} MemoryRegionType;

// Phase 5: 10-feature vector struct (Paper Table 2)
// 10 features extracted from the paper's 6 signals
typedef struct _FeatureVector {
	// Signal 1: Shannon Entropy (F1)
	double F1_entropy;              // [0, 8] bits/byte

	// Signal 2: Chi-square Uniformity (F2)
	double F2_chi_square;           // [0, ∞]

	// Signal 3: Length Constraints (F3, F4, F5)
	size_t F3_length;               // [1, 1024] bytes
	int F4_is_standard_key_length;  // {0, 1} - 8, 16, 24, 32 bytes (DES, AES-128/192/256)
	int F5_is_standard_iv_length;   // {0, 1} - 8, 12, 16 bytes

	// Signal 4: Memory Region Type (F6)
	MemoryRegionType F6_region_type; // [0, 3]

	// Signal 5: Temporal Change Pattern (F7, F8)
	int F7_change_count;            // [0, N] change count across snapshots
	TemporalPatternType F8_pattern_type; // [0, 3]

	// Signal 6: Cross-Signal Synergy (F9, F10)
	double F9_entropy_length_interaction; // F1 × log2(F3 + 1)
	int F10_high_confidence_key;    // {0, 1} - F1 >= 7.5 AND F4 == 1
} FeatureVector;

// Phase 5: Classification result struct (hybrid ensemble output)
// Output format of Paper Algorithm 1
typedef struct _ClassificationResult {
	CryptoClassification predicted_class;  // Predicted class (0-4)
	double confidence;                      // Confidence [0, 1]
	int heuristic_score;                    // Heuristic score [0, 100]
	double class_probabilities[5];          // Per-class probabilities (for future ML extension)
	char class_name[32];                    // Class name string
} ClassificationResult;

// ============================================================
// Utility Macros
// ============================================================

// Safe memory deallocation macro
// Checks for NULL, frees memory, and sets pointer to NULL to prevent double-free
#define SAFE_FREE(ptr) { \
	if (ptr) { \
		free(ptr); \
		ptr = NULL; \
	} \
}


// ============================================================
// Function Prototype Declarations
// ============================================================

// --- Phase 1: Process and Module Management Functions ---
ModuleInfo2 GetModuleInfo(const char* targetDll);  // Retrieve target DLL's process ID and base address
BOOL IsModuleLoaded(DWORD processID, LPCWSTR moduleName);  // Check if a module is loaded in a specific process
LPVOID GetDllBaseAddress(DWORD processId, const char* dllName);  // Get the base address of a DLL

// --- Utility Functions ---
void ConvertToLowerCase(char* dest, const wchar_t* src, size_t maxLen);  // Convert wide string to lowercase ASCII
void SafeStringToLower(const TCHAR* input, char* output, size_t outputSize);  // Safely convert TCHAR string to lowercase // 24.11.11
DWORD GetThreadIdForAddress(HANDLE processHandle, SIZE_T address);  // Retrieve thread ID owning a specific memory address

// --- Phase 2-3: Memory Snapshot and Comparison Functions ---
void ReadMemoryAndSaveToFile_case22(HANDLE processHandle, const char* filename, const char* dllName, DWORD processId);  // Save process memory as a snapshot
void CompareFilesAndLogDifferences_case22(HANDLE processHandle, const char* file1, const char* file2, const char* logFile);  // Compare two snapshot files and log differences
void LogMemoryChange(HANDLE processHandle, FILE* log, MemoryChange* change, const char* regionType, MemoryChangeMap* changeMap);  // Record memory changes to a log file

// --- Phase 4: Memory Change Map Management Functions ---
MemoryChangeMap* InitializeChangeMap(int capacity);  // Initialize memory change map (dynamic allocation)
void FreeChangeMap(MemoryChangeMap* map);  // Free memory change map (prevent memory leaks)
void AddAddressHistory(MemoryChangeMap* map, SIZE_T address, unsigned char value, int snapshot_idx, const char* region_type, int* last_idx_hint);  // Add snapshot value for a specific address
ChangePattern AnalyzeChangePattern(const AddressChangeHistory* history);  // Analyze address change history and classify pattern
void GenerateChangeLog(MemoryChangeMap* map, const char* output_file);  // Output change pattern analysis results to log file
ChangePattern* GetPatternFromChangeMap(MemoryChangeMap* map, SIZE_T address);  // [Phase 4-5 Integration] Query pattern for a specific address

// --- Phase 5: Cryptographic Key Detection and Classification Functions ---
double calculate_entropy(const unsigned char* data, size_t length);  // Shannon entropy calculation (data randomness measurement)
double calculate_chi_square(const unsigned char* data, size_t length);  // Chi-square test (uniform distribution check)
bool is_uniform_distribution(double chi_square);  // Determine uniform distribution based on chi-square value
int is_crypto_symbol(const char* symbolName);  // Check if symbol name is crypto-related (0: no, 1: partial match, 2: exact match)
bool is_standard_key_length(size_t length);  // Check if standard cryptographic key length (8, 16, 24, 32 bytes, etc.)
bool is_standard_iv_length(size_t length);  // Check if standard IV (initialization vector) length (8, 12, 16 bytes)
bool is_potential_crypto_data(const unsigned char* data, size_t length);  // Heuristic analysis to determine if data appears encrypted
int CalculateClassificationScore_Phase5(const unsigned char* data, size_t length, const char* region_type, const ChangePattern* pattern);  // Multi-signal based classification score calculation (0-100 points)
double get_entropy_threshold_for_confidence(int confidence_level);  // Return entropy threshold by confidence level (statistics-based)

// --- Phase 5: ML-based Multi-Class Classification Functions (Paper Section 3) ---
FeatureVector ExtractFeatureVector(const unsigned char* data, size_t length, const char* region_type, const ChangePattern* pattern);  // Extract 10-feature vector
MemoryRegionType ParseRegionType(const char* region_type_str);  // Convert string region type to enum
TemporalPatternType GetTemporalPatternType(const ChangePattern* pattern);  // Temporal pattern classification
ClassificationResult ClassifyMemoryBlock(const unsigned char* data, size_t length, const char* region_type, const ChangePattern* pattern);  // Hybrid ensemble classification (Paper Algorithm 1)
const char* GetClassName(CryptoClassification cls);  // Convert class enum to string

// --- Phase 5: Multi-Thread Tracking Functions ---
int find_all_threads_for_address(HANDLE processHandle, SIZE_T address, DWORD* threads_out, int max_threads);  // Find all threads accessing a specific memory address

// ============================================================
// qsort comparison function: used for sorting by address
// ============================================================
// Comparison function for sorting AddressChangeHistory structs in ascending address order
// Used in Phase 5 Spatial Analysis to find contiguous memory blocks
int compare_history_address(const void* a, const void* b) {
	const AddressChangeHistory* ha = (const AddressChangeHistory*)a;
	const AddressChangeHistory* hb = (const AddressChangeHistory*)b;
	if (ha->address < hb->address) return -1;  // return -1 if a < b
	if (ha->address > hb->address) return 1;   // return 1 if a > b
	return 0;  // return 0 if equal
}


// ============================================================
// main function: program entry point
// ============================================================
int main() {
	// ============================================================
	// Initialization: Console encoding setup
	// ============================================================
	//setlocale(LC_ALL, "ko_KR.UTF-8");  // UTF-8 locale setting (commented out)
	SetConsoleOutputCP(65001);  // Set console output to UTF-8 for Unicode support
	SetConsoleCP(65001);        // Set console input to UTF-8
	wprintf(L"Unicode Test\n");  // Unicode output test

	// ============================================================
	// Variable declarations and initialization
	// ============================================================
	// Open log file (currently unused)
	FILE* fp;
	errno_t err = fopen_s(&fp, "log.txt", "w");

	// Target process and DLL information variables
	char exeName[MAX_PATH] = { 0x00, };  // Target executable file name (.exe)
	char dllName[MAX_PATH] = { 0x00, };  // Target DLL name

	DWORD processId = NULL;  // Target process ID
	DllInfo dllInfo;         // DLL information struct // 23.06.17 By KJ

	// Windows API handles declaration
	HANDLE snapshotHandle = NULL;        // Process snapshot handle
	BOOL found = FALSE;                  // Target process search success flag
	HANDLE targetProcessHandle = NULL;   // Target process handle
	HANDLE moduleSnapshotHandle = NULL;  // Module snapshot handle

	// Structs for process and module enumeration
	PROCESSENTRY32 processEntry;  // Process information storage
	MODULEENTRY32 moduleEntry;    // Module (DLL) information storage

	// Menu system related variables
	int Menu = 0;                    // User-selected menu number
	HANDLE targetHeap = NULL;        // Target heap handle (currently unused)
	LPVOID targetAddress = NULL;     // Target memory address (currently unused)
	SIZE_T targetSize = 0;           // Target memory size (currently unused)
	char tmp[MAX_PATH] = { 0x00, };  // Temporary buffer

	// Process creation related structs (currently unused) //23.07.28
	STARTUPINFO si = { sizeof(si) };
	PROCESS_INFORMATION pi;
	HANDLE processHandle;
	LPVOID remoteStackAddress;
	SIZE_T stackSize;
	DWORD size;
	TCHAR buffer[MAX_PATH] = { 0x00, };
	HMODULE hModule;

	// ============================================================
	// Main loop: Menu system
	// ============================================================
	while (1)
	{
	MENU_TOP:  // Label for goto return (return to menu)
		// ============================================================
		// Menu display
		// ============================================================
		printf("\n#################################################################\n");
		printf("\n");
		printf("  0) Initialize App(Enter Target Name(.exe and .dll)) \n");  // Initialize target process and DLL
		printf("  1) Change Target Name(.exe and .dll) \n");                 // Change target
		printf("\n");
		printf("  22) [ENHANCED] 5-Phase Integrated Memory Analysis \n");    // 5-phase integrated memory analysis (core feature)
		printf("      - Phase 1: Process & DLL Discovery \n");               // Process and DLL discovery
		printf("      - Phase 2: Selective Snapshot (5 snapshots) \n");      // Selective snapshot creation
		printf("      - Phase 3: Differential Comparison \n");               // Differential comparison
		printf("      - Phase 4: Address-Level Change Tracking \n");         // Address-level change tracking
		printf("      - Phase 5: Multi-Signal Cryptographic Classifier \n"); // Multi-signal cryptographic classification
		printf("\n");
		printf("  23) [NEW] Phase 6: API Hooking Injection (Detours) \n");   // API hooking injection
		printf("\n");
		printf("  99) Exit. \n");  // Exit
		printf("\n#################################################################\n");
		printf("\n");
		printf(" Enter Menu: ");
		scanf_s("%d", &Menu);  // Read menu selection input

		// ============================================================
		// Menu handling: switch statement
		// ============================================================
		switch (Menu) {
		// ============================================================
		// Case 0: App initialization - Target process and DLL monitoring
		// ============================================================
		// Monitors for target process and DLL loading, and upon detection
		// acquires the process ID and related information // 23.12.29
		case 0:
		{
			// ====================================================
			// User input: Target executable and DLL names
			// ====================================================
			printf("\n###################################################\n");
			printf("Enter the execution file name: ");
			getchar(); // Flush remaining newline from buffer (handle previous scanf_s newline)
			fgets(exeName, sizeof(exeName), stdin);  // Read executable file name
			exeName[strcspn(exeName, "\n")] = '\0';  // Remove trailing newline from fgets input

			printf("Enter the Target DLL name: ");
			//getchar(); // Flush remaining newline from buffer (commented out)
			fgets(dllName, sizeof(dllName), stdin);  // Read DLL name
			dllName[strcspn(dllName, "\n")] = '\0';  // Remove trailing newline
			printf("\n###################################################\n");

			// Display the entered information
			printf("Executable Name: %s\n", exeName);
			printf("DLL Name: %s\n", dllName);


			char unicodeDllName[MAX_PATH] = { 0x00, };
			MultiByteToWideChar(CP_ACP, 0, dllName, -1, (LPWSTR)unicodeDllName, MAX_PATH);


			// Wait for DLL to be loaded in memory
			while (TRUE) {
				// Stop waiting if ESC key is pressed
				if (_kbhit()) {
					int ch = _getch();
					if (ch == 27) {
						printf("ESC key pressed. Monitoring cancelled.\n");
						break;
					}
				}

				// Create process snapshot handle
				snapshotHandle = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
				if (snapshotHandle == INVALID_HANDLE_VALUE) {
					printf("Failed to create process snapshot.\n");
					continue;
				}

				processEntry.dwSize = sizeof(PROCESSENTRY32);

				// Refresh process snapshot
				if (Process32First(snapshotHandle, &processEntry)) {
					do {
						char lowercaseProcessName[MAX_NAME_LENGTH] = { 0x00, };

						ConvertToLowerCase(lowercaseProcessName, processEntry.szExeFile, MAX_NAME_LENGTH);

						if (strlen(lowercaseProcessName) == strlen(exeName)) {
							bool isEqual = true;
							for (int i = 0; exeName[i] != '\0'; i++) {
								if (tolower(exeName[i]) != lowercaseProcessName[i]) {
									isEqual = false;
									break;
								}
							}

							if (isEqual) {
								// Open target process handle
								targetProcessHandle = OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION | PROCESS_ALL_ACCESS, FALSE, processEntry.th32ProcessID);
								if (targetProcessHandle != NULL) {
									// Create module snapshot handle
									moduleSnapshotHandle = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, processEntry.th32ProcessID);
									if (moduleSnapshotHandle != INVALID_HANDLE_VALUE) {
										// Check DLL loading
										if (IsModuleLoaded(processEntry.th32ProcessID, (LPCWSTR)unicodeDllName)) {
											printf("Module %s is loaded in process %s.\n", dllName, exeName);
											//CloseHandle(moduleSnapshotHandle);
											//CloseHandle(targetProcessHandle);
											goto FoundModule;
										}
										CloseHandle(moduleSnapshotHandle);
									}

								}
								else
									CloseHandle(targetProcessHandle);
							}
						}
					} while (Process32Next(snapshotHandle, &processEntry));
				}
				else {
					printf("No processes found.\n");
				}

				Sleep(100); // Brief wait to reduce CPU usage
			}

		FoundModule:
			CloseHandle(snapshotHandle);

			// Call the function to get module information
			ModuleInfo2 result = GetModuleInfo(dllName);

			// Display the result
			if (result.processId != 0) {
				printf(" [DEBUG} Target DLL found in process ID: %u, targetProcessHandle : [%02X]\n", result.processId, targetProcessHandle);
				printf(" [DEBUG} Base address in virtual memory: 0x%p\n", result.moduleBase);
			}
			else {
				printf(" Target DLL not found in the specified process.\n");
			}

		}
		break;
		case 1:
		{
			// Re-acquire process ID using target process name and DLL name
			// Initialization
			memset(exeName, 0x00, sizeof(exeName));
			memset(dllName, 0x00, sizeof(dllName));

			// Re-input
			printf("\n###################################################\n");
			printf(" >> Enter the execution file name: ");
			getchar(); // Flush remaining newline from buffer
			fgets(exeName, sizeof(exeName), stdin);
			exeName[strcspn(exeName, "\n")] = '\0'; // Remove trailing newline from fgets input

			printf(" >> Enter the Target DLL name: ");
			//getchar(); // Flush remaining newline from buffer
			fgets(dllName, sizeof(dllName), stdin);
			dllName[strcspn(dllName, "\n")] = '\0'; // Remove trailing newline from fgets input
			printf("\n###################################################\n");

			// Get process handle
			snapshotHandle = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
			if (snapshotHandle == INVALID_HANDLE_VALUE) {
				printf("Failed to create process snapshot.\n");
				printf("\nFailed to acquire target process handle information!\n\n");
				return 1;
			}


			processEntry.dwSize = sizeof(PROCESSENTRY32);

			// Find target process
			found = FALSE;
			if (Process32First(snapshotHandle, &processEntry)) {
				do {
					// Convert all characters to lowercase for comparison
					char lowercaseProcessName[MAX_PATH];
					for (int i = 0; i < MAX_PATH; i++) {
						lowercaseProcessName[i] = tolower(processEntry.szExeFile[i]);
					}

					if (_stricmp(lowercaseProcessName, exeName) == 0) {
						found = TRUE;
						break;
					}
				} while (Process32Next(snapshotHandle, &processEntry));
			}

			// Exit if target process was not found
			if (!found) {
				printf("Target process not found.\n");
				CloseHandle(snapshotHandle);
				//return 1;
				goto MENU_TOP;
			}

			// Open target process handle
			targetProcessHandle = OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION | PROCESS_ALL_ACCESS, FALSE, processEntry.th32ProcessID);
			printf(" [DEBUG] OpenProcess( ), processEntry.th32ProcessID = [%d] \n", processEntry.th32ProcessID);
			if (targetProcessHandle == NULL) {
				printf("Failed to open target process.\n");
				CloseHandle(snapshotHandle);
				return 1;
			}

			// Create module snapshot handle
			moduleSnapshotHandle = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, processEntry.th32ProcessID);
			printf(" [DEBUG] CreateToolhelp32Snapshot( ), processEntry.th32ProcessID = [%d] \n", processEntry.th32ProcessID);

			if (moduleSnapshotHandle == INVALID_HANDLE_VALUE) {
				printf("Failed to create module snapshot.\n");
				CloseHandle(targetProcessHandle);
				CloseHandle(snapshotHandle);
				return 1;
			}

			moduleEntry.dwSize = sizeof(MODULEENTRY32);

			// Find first module in target process
			if (!Module32First(moduleSnapshotHandle, &moduleEntry)) {
				printf("Failed to find first module.\n");

				CloseHandle(moduleSnapshotHandle);
				CloseHandle(targetProcessHandle);
				CloseHandle(snapshotHandle);
				return 1;
			}

			// Call the function to get module information
			ModuleInfo2 result = GetModuleInfo(dllName);

			// Display the result
			if (result.processId != 0) {
				printf(" [DEBUG} Target DLL found in process ID: %u\n", result.processId);
				printf(" [DEBUG} Base address in virtual memory: 0x%p\n", result.moduleBase);
			}
			else {
				printf(" Target DLL not found in the specified process.\n");
			}

		}
		break;
		case 22:
			printf(" - Target Process: %s, process ID : %d \n", exeName, processEntry.th32ProcessID);
			processId = processEntry.th32ProcessID;
			{
				// [FIX] Declare ALL variables early for goto scope
				// Prevent variable initialization skip errors due to goto CLEANUP_PHASE
				MemoryChangeMap* changeMap = NULL;
				FILE* phase5_log = NULL;

				// Snapshot configuration variables
				int snapshot_count = DEFAULT_SNAPSHOTS;
				int snapshot_interval_ms = 1000;
				int user_input = 0;
				int interval_choice = 0;
				int ch = 0;

				// Sensitivity related variables
				int sensitivity_threshold = 60;
				int short_block_threshold = 70;
				int sens_choice = 0;
				int ch_sens = 0;

				// Filename arrays (declared with max size since C++ does not support VLA)
				char snapshotFilenames[MAX_SNAPSHOTS_CAPACITY][50];
				char prevSnapshotFilename[50];
				char logFilename[50];

				printf("\n========================================\n");
				printf("Phase 1-5: Integrated Memory Analysis\n");
				printf("========================================\n\n");

				// [FIX] Prompt user for input if dllName is not set
				if (strlen(dllName) == 0) {
					printf(" [WARNING] Target DLL name is not set.\n");
					printf(" Please enter the target DLL name (e.g., crypto.dll): ");
					fgets(dllName, sizeof(dllName), stdin);
					dllName[strcspn(dllName, "\n")] = '\0';
					if (strlen(dllName) == 0) {
						fprintf(stderr, " [ERROR] DLL name cannot be empty.\n");
						goto MENU_TOP;
					}
					printf(" [DEBUG] Target DLL set to: %s\n", dllName);
				}

				printf(" [DEBUG] Opening process ID: %d\n", processId);
				// [FIX] Added PROCESS_SUSPEND_RESUME permission (required for NtSuspendProcess/NtResumeProcess)
				HANDLE processHandle = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | PROCESS_SUSPEND_RESUME, FALSE, processId);
				if (processHandle == NULL) {
					DWORD error = GetLastError();
					fprintf(stderr, " [ERROR] Failed to open process ID %d. Error: %lu\n", processId, error);
					goto MENU_TOP;
				}

				// [IMPROVED] Phase 1-3: User-configurable snapshot count support
				// Variables are declared at the top of the block (goto scope issue resolved)

				printf("\n--- Snapshot Configuration ---\n");
				printf("Enter number of snapshots (default %d, max %d): ", DEFAULT_SNAPSHOTS, MAX_SNAPSHOTS_CAPACITY);
				if (scanf("%d", &user_input) == 1 && user_input >= 2 && user_input <= MAX_SNAPSHOTS_CAPACITY) {
					snapshot_count = user_input;
					printf(" [DEBUG] Using %d snapshots\n", snapshot_count);
				} else {
					printf(" [DEBUG] Invalid input, using default %d snapshots\n", DEFAULT_SNAPSHOTS);
					snapshot_count = DEFAULT_SNAPSHOTS;
				}
				// [FIX] Completely flush input buffer (both success/failure cases)
				// Prevent leftover characters from scanf being falsely detected by _kbhit()
				while ((ch = getchar()) != '\n' && ch != EOF);

				// [NEW] Snapshot interval setting - fast mode support for capturing short-lived keys
				printf("\nSelect snapshot interval mode:\n");
				printf("  1. Fast mode     (100ms) - Best for short-lived keys\n");
				printf("  2. Moderate mode (200ms) - Balanced detection\n");
				printf("  3. Normal mode   (500ms) - Standard analysis\n");
				printf("  4. Slow mode    (1000ms) - Default, low overhead\n");
				printf("Enter choice (1-4, default 4): ");
				if (scanf("%d", &interval_choice) == 1) {
					switch (interval_choice) {
						case 1: snapshot_interval_ms = 100; break;
						case 2: snapshot_interval_ms = 200; break;
						case 3: snapshot_interval_ms = 500; break;
						case 4: snapshot_interval_ms = 1000; break;
						default: snapshot_interval_ms = 1000; break;
					}
				}
				printf(" [DEBUG] Snapshot interval: %dms\n", snapshot_interval_ms);

				// [FIX] Flush input buffer
				while ((ch = getchar()) != '\n' && ch != EOF);

				printf("\nSnapshot interval: %dms\n", snapshot_interval_ms);
				printf("Total analysis time: ~%.1f seconds\n\n",
					(snapshot_count * snapshot_interval_ms) / 1000.0);

				// Phase 1-3: Create N snapshots and perform differential comparison

				printf("\n--- Phase 2: Selective Snapshot Creation ---\n");
				printf(" [INFO] Press ESC to stop analysis at any time.\n\n");

				// [FIX] Completely flush keyboard buffer (remove leftover input)
				while (_kbhit()) _getch();

				for (int i = 0; i < snapshot_count; i++) {
					// [FIX] Check for user interruption - consume all pending keys
					while (_kbhit()) {
						int key = _getch();
						if (key == 27) { // ESC key
							printf("\n [User Interrupt] ESC pressed. Stopping analysis...\n");
							goto CLEANUP_PHASE;
						}
						// Ignore non-ESC keys and continue
					}

					sprintf_s(snapshotFilenames[i], sizeof(snapshotFilenames[i]),
						"snapshot_case22_%d.bin", i);
					printf(" [DEBUG] Creating snapshot %d/%d: %s\n",
						i + 1, snapshot_count, snapshotFilenames[i]);
					ReadMemoryAndSaveToFile_case22(processHandle, snapshotFilenames[i], dllName, processId);

					// Phase 3: Differential comparison (except the first)
					if (i > 0) {
						sprintf_s(logFilename, sizeof(logFilename),
							"phase3_differences_%d.log", i);
						printf(" [DEBUG] Phase 3: Comparing snapshots %d and %d -> %s\n",
							i - 1, i, logFilename);
						CompareFilesAndLogDifferences_case22(processHandle,
							snapshotFilenames[i - 1], snapshotFilenames[i], logFilename);
					}

					// [IMPROVED] Wait between snapshots - dynamic interval, split into 10ms chunks for ESC key checking
					if (i < snapshot_count - 1) {
						int wait_iterations = snapshot_interval_ms / 10;  // Split into 10ms intervals
						for (int wait = 0; wait < wait_iterations; wait++) {
							Sleep(10);
							// Check for ESC during wait
							if (_kbhit()) {
								int key = _getch();
								if (key == 27) {
									printf("\n [User Interrupt] ESC pressed. Stopping analysis...\n");
									goto CLEANUP_PHASE;
								}
								// Ignore non-ESC keys and continue
							}
						}
					}
				}

				printf("\n--- Phase 4: Address-Level Change Tracking ---\n");
				printf(" [DEBUG] Initializing memory change map...\n");

				// Phase 4: Initialize memory change map
				changeMap = InitializeChangeMap(1000000); // Increased initial capacity
				if (!changeMap) {
					fprintf(stderr, " [ERROR] Failed to initialize change map\n");
					CloseHandle(processHandle);
					goto MENU_TOP;
				}

				// Parse all snapshots and build per-address history
				for (int i = 0; i < snapshot_count; i++) {
					// [FIX] Check for user interruption - consume all pending keys
					while (_kbhit()) {
						int key = _getch();
						if (key == 27) { // ESC key
							printf("\n [User Interrupt] ESC pressed. Stopping analysis...\n");
							goto CLEANUP_PHASE;
						}
					}

					printf(" [DEBUG] Parsing snapshot %d: %s\n", i, snapshotFilenames[i]);

					FILE* snapshot_file = NULL;
					if (fopen_s(&snapshot_file, snapshotFilenames[i], "rb") != 0) {
						fprintf(stderr, " [ERROR] Cannot open snapshot file %s\n", snapshotFilenames[i]);
						continue;
					}

					// Parse snapshot file
					MemoryRegionHeader header;
					SIZE_T bytesRead;
					unsigned char* buffer = NULL;
					int regions_processed = 0;
					
					// [OPTIMIZATION] Hint for sequential access
					int last_idx_hint = 0;

					while (1) {
						// Read header
						if (fread(&header, sizeof(MemoryRegionHeader), 1, snapshot_file) != 1) {
							break;
						}

						// Read data size
						if (fread(&bytesRead, sizeof(SIZE_T), 1, snapshot_file) != 1) {
							break;
						}

						if (bytesRead == 0) continue;

						// Allocate buffer
						buffer = (unsigned char*)malloc(bytesRead);
						if (!buffer) continue;

						// Read data
						if (fread(buffer, 1, bytesRead, snapshot_file) != bytesRead) {
							SAFE_FREE(buffer);
							continue;
						}

						// Add each byte to the memory map
						for (SIZE_T j = 0; j < bytesRead; j++) {
							SIZE_T current_address = header.baseAddress + j;
							unsigned char value = buffer[j];
							AddAddressHistory(changeMap, current_address, value, i, header.regionType, &last_idx_hint);
						}

						regions_processed++;
						SAFE_FREE(buffer);
					}

					fclose(snapshot_file);
					printf(" [DEBUG] Processed %d regions from snapshot %d\n", regions_processed, i);
				}

				// Phase 4: Generate change log
				printf(" [DEBUG] Generating Phase 4 change tracking log...\n");
				GenerateChangeLog(changeMap, "phase4_change_tracking.log");

				// Phase 5: Generate final classification log
				printf("\n--- Phase 5: Cryptographic Classification ---\n");

				// [NEW] Sensitivity setting - threshold adjustment to reduce false negatives
				// Variables declared at the top of the block (goto scope issue resolved)
				printf("\nSelect detection sensitivity:\n");
				printf("  1. High sensitivity   (40pt) - More candidates, higher false positives\n");
				printf("  2. Normal sensitivity (60pt) - Balanced (default)\n");
				printf("  3. Low sensitivity    (75pt) - Fewer candidates, higher precision\n");
				printf("Enter choice (1-3, default 2): ");
				if (scanf("%d", &sens_choice) == 1) {
					switch (sens_choice) {
						case 1:
							sensitivity_threshold = 40;
							short_block_threshold = 55;
							break;
						case 2:
							sensitivity_threshold = 60;
							short_block_threshold = 70;
							break;
						case 3:
							sensitivity_threshold = 75;
							short_block_threshold = 85;
							break;
						default:
							sensitivity_threshold = 60;
							short_block_threshold = 70;
							break;
					}
				}
				printf(" [DEBUG] Detection threshold: %d (short blocks: %d)\n",
					sensitivity_threshold, short_block_threshold);

				// [FIX] Flush input buffer
				while ((ch_sens = getchar()) != '\n' && ch_sens != EOF);

				if (fopen_s(&phase5_log, "phase5_classification.log", "w") == 0) {
					fprintf(phase5_log, "=========================================\n");
					fprintf(phase5_log, "Phase 5: Final Classification Report\n");
					fprintf(phase5_log, "Sensitivity Threshold: %d (Short blocks: %d)\n",
						sensitivity_threshold, short_block_threshold);
					fprintf(phase5_log, "=========================================\n\n");

					int high_confidence_count = 0;
					int medium_confidence_count = 0;
					int low_confidence_count = 0;

					// [FIX] 3. Sort by address for Spatial Analysis
					qsort(changeMap->histories, changeMap->history_count, sizeof(AddressChangeHistory), compare_history_address);

					// [FIX] 4. Spatial Analysis (Block-based) instead of Temporal
					// Iterate through sorted histories to find contiguous high-entropy blocks
					int i = 0;
					while (i < changeMap->history_count) {
						// [FIX] Check for user interruption - consume all pending keys
						while (_kbhit()) {
							int key = _getch();
							if (key == 27) { // ESC key
								printf("\n [User Interrupt] ESC pressed. Stopping analysis...\n");
								goto CLEANUP_PHASE;
							}
						}

						// Check for contiguous block (try to form up to 32 bytes)
						int block_len = 0;
						unsigned char block_data[32];
						AddressChangeHistory* start_node = &changeMap->histories[i];

						// Collect contiguous bytes
						for (int j = 0; j < 32; j++) {
							if (i + j >= changeMap->history_count) break;

							AddressChangeHistory* current = &changeMap->histories[i + j];
							// Check address continuity
							if (current->address != start_node->address + j) break;

							// Use the latest snapshot value for spatial analysis
							int last_idx = current->snapshot_count > 0 ? current->snapshot_count - 1 : 0;
							block_data[j] = current->values[last_idx];
							block_len++;
						}

						// [IMPROVED] Analyze if we have a meaningful block (8 bytes minimum for DES/3DES keys)
						// 8-15 bytes: Apply stricter entropy check (detect short keys like DES, Blowfish)
						// 16+ bytes: Maintain existing thresholds (AES, RSA, etc.)
						if (block_len >= 8) {
							// Use the start node's temporal pattern as a reference (e.g., is it static or dynamic?)
							ChangePattern pattern = AnalyzeChangePattern(start_node);

							// [NEW] ML-based 5-class classification (Paper Section 3)
							ClassificationResult result = ClassifyMemoryBlock(block_data, block_len, start_node->region_type, &pattern);

							// [NEW] Feature vector extraction (Paper Table 2)
							FeatureVector fv = ExtractFeatureVector(block_data, block_len, start_node->region_type, &pattern);

							// [IMPROVED] Apply dynamic threshold (using sensitivity_threshold)
							// Short blocks (8-15 bytes) require higher entropy
							// False positive prevention: short blocks have higher chance of random high entropy
							int min_score_threshold = sensitivity_threshold;  // Apply dynamic threshold
							if (block_len < 16) {
								// Short blocks: entropy condition + dynamic threshold applied
								// Higher sensitivity (lower threshold) relaxes entropy condition
								double min_entropy = (sensitivity_threshold <= 40) ? 6.5 : 7.0;
								if (fv.F1_entropy < min_entropy || result.heuristic_score < short_block_threshold) {
									i++;
									continue;
								}
								min_score_threshold = short_block_threshold;  // Short block dynamic threshold
							}

							// Filter candidates
							if (result.heuristic_score >= min_score_threshold) {
								fprintf(phase5_log, "\n=== High Priority Block ===\n");
								fprintf(phase5_log, "Start Address: 0x%08X\n", (unsigned int)start_node->address);
								fprintf(phase5_log, "Length: %d bytes\n", block_len);
								fprintf(phase5_log, "Region: %s\n", start_node->region_type);

								// [NEW] Output 5-class classification result
								fprintf(phase5_log, "\n--- ML Classification (5-Class) ---\n");
								fprintf(phase5_log, "Predicted Class: %s\n", result.class_name);
								fprintf(phase5_log, "Confidence: %.2f%%\n", result.confidence * 100.0);
								fprintf(phase5_log, "Heuristic Score: %d/100\n", result.heuristic_score);
								fprintf(phase5_log, "Class Probabilities:\n");
								fprintf(phase5_log, "  KEY:        %.2f%%\n", result.class_probabilities[CRYPTO_CLASS_KEY] * 100.0);
								fprintf(phase5_log, "  IV:         %.2f%%\n", result.class_probabilities[CRYPTO_CLASS_IV] * 100.0);
								fprintf(phase5_log, "  CIPHERTEXT: %.2f%%\n", result.class_probabilities[CRYPTO_CLASS_CIPHERTEXT] * 100.0);
								fprintf(phase5_log, "  PLAINTEXT:  %.2f%%\n", result.class_probabilities[CRYPTO_CLASS_PLAINTEXT] * 100.0);
								fprintf(phase5_log, "  NON_CRYPTO: %.2f%%\n", result.class_probabilities[CRYPTO_CLASS_NON_CRYPTO] * 100.0);

								// [NEW] Output 10-feature vector (Paper Table 2)
								fprintf(phase5_log, "\n--- Feature Vector (10 Features) ---\n");
								fprintf(phase5_log, "F1 (Entropy):     %.4f bits/byte\n", fv.F1_entropy);
								fprintf(phase5_log, "F2 (Chi-Square):  %.2f\n", fv.F2_chi_square);
								fprintf(phase5_log, "F3 (Length):      %zu bytes\n", fv.F3_length);
								fprintf(phase5_log, "F4 (Std Key Len): %d\n", fv.F4_is_standard_key_length);
								fprintf(phase5_log, "F5 (Std IV Len):  %d\n", fv.F5_is_standard_iv_length);
								fprintf(phase5_log, "F6 (Region Type): %d\n", fv.F6_region_type);
								fprintf(phase5_log, "F7 (Change Count):%d\n", fv.F7_change_count);
								fprintf(phase5_log, "F8 (Pattern Type):%d\n", fv.F8_pattern_type);
								fprintf(phase5_log, "F9 (Synergy):     %.4f\n", fv.F9_entropy_length_interaction);
								fprintf(phase5_log, "F10 (High Conf):  %d\n", fv.F10_high_confidence_key);
								fprintf(phase5_log, "Pattern: %s\n", pattern.pattern_name);

								// Hex dump
								fprintf(phase5_log, "\nData: ");
								for (int k = 0; k < block_len; k++) fprintf(phase5_log, "%02X ", block_data[k]);
								fprintf(phase5_log, "\n");

								if (result.heuristic_score >= 90) high_confidence_count++;
								else if (result.heuristic_score >= 75) medium_confidence_count++;
								else low_confidence_count++;
							}
						}

						// Move to next address (Sliding window of 1 byte)
						// Note: To improve performance, we could jump 'block_len' if a key is found,
						// but sliding by 1 ensures we find keys at any alignment.
						i++;
					}

					fprintf(phase5_log, "\n=========================================\n");
					fprintf(phase5_log, "Summary (Paper Section 5: Results):\n");
					fprintf(phase5_log, "  High Confidence (90+): %d\n", high_confidence_count);
					fprintf(phase5_log, "  Medium Confidence (75-89): %d\n", medium_confidence_count);
					fprintf(phase5_log, "  Low Confidence (60-74): %d\n", low_confidence_count);
					fprintf(phase5_log, "  Total Candidates: %d\n", high_confidence_count + medium_confidence_count + low_confidence_count);
					fprintf(phase5_log, "=========================================\n");

					fclose(phase5_log);
					printf(" [DEBUG] Phase 5 classification log written to phase5_classification.log\n");
				}

				// Memory cleanup
			CLEANUP_PHASE:
				// [FIX] Close phase5_log if it's open (prevent file handle leak when interrupted by ESC)
				if (phase5_log) {
					fclose(phase5_log);
					phase5_log = NULL;
				}
				FreeChangeMap(changeMap);
				CloseHandle(processHandle);

				printf("\n========================================\n");
				printf("Analysis Complete!\n");
				printf("Output Files:\n");
				printf("  - snapshot_case22_0.bin ~ 4.bin\n");
				printf("  - phase3_differences_1.log ~ 4.log\n");
				printf("  - phase4_change_tracking.log\n");
				printf("  - phase5_classification.log\n");
				printf("========================================\n\n");
			}
			break;

		case 23:
		{
			printf("\n========================================\n");
			printf("Phase 6: API Hooking Injection (Detours)\n");
			printf("========================================\n\n");

			// [FIX] PID Re-validation
			if (processId == 0) {
				printf(" [WARNING] Target process ID is 0.\n");
				printf(" Please enter the Target Process ID: ");
				scanf_s("%d", &processId);
				getchar(); // Consume newline
			}

			// [NEW] Write Target DLL Name to Config File for Monitor_DLL
			FILE* cfgFunc = fopen("target_dll_name.txt", "w");
			if (cfgFunc) {
				fprintf(cfgFunc, "%s", dllName);
				fclose(cfgFunc);
				printf(" [CONFIG] Target DLL name '%s' written to config file.\n", dllName);
			} else {
				fprintf(stderr, " [WARNING] Failed to write config file. Monitor_DLL might use default name.\n");
			}

			printf("Target Process ID: %d (%s)\n", processId, exeName);

			char dllPath[MAX_PATH];
			// Assuming Monitor_DLL.dll is in the same directory as the analyzer
			GetModuleFileNameA(NULL, dllPath, MAX_PATH);
			PathRemoveFileSpecA(dllPath);
			PathAppendA(dllPath, "Monitor_DLL.dll");

			printf("Injecting DLL: %s\n", dllPath);

			// Check if DLL exists
			if (GetFileAttributesA(dllPath) == INVALID_FILE_ATTRIBUTES) {
				fprintf(stderr, " [ERROR] Monitor_DLL.dll not found at %s\n", dllPath);
				fprintf(stderr, "         Please compile Monitor_DLL.c first.\n");
				break;
			}

			HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, processId);
			if (!hProcess) {
				fprintf(stderr, " [ERROR] Failed to open process. Error: %lu\n", GetLastError());
				break;
			}

			// 1. Allocate memory in target process for DLL path
			LPVOID pRemoteBuf = VirtualAllocEx(hProcess, NULL, strlen(dllPath) + 1, MEM_COMMIT, PAGE_READWRITE);
			if (!pRemoteBuf) {
				fprintf(stderr, " [ERROR] VirtualAllocEx failed.\n");
				CloseHandle(hProcess);
				break;
			}

			// 2. Write DLL path to target process memory
			if (!WriteProcessMemory(hProcess, pRemoteBuf, (LPVOID)dllPath, strlen(dllPath) + 1, NULL)) {
				fprintf(stderr, " [ERROR] WriteProcessMemory failed.\n");
				VirtualFreeEx(hProcess, pRemoteBuf, 0, MEM_RELEASE);
				CloseHandle(hProcess);
				break;
			}

			// 3. Create remote thread to load the DLL
			HANDLE hThread = CreateRemoteThread(hProcess, NULL, 0,
				(LPTHREAD_START_ROUTINE)GetProcAddress(GetModuleHandleA("Kernel32"), "LoadLibraryA"),
				pRemoteBuf, 0, NULL);

			if (!hThread) {
				fprintf(stderr, " [ERROR] CreateRemoteThread failed. Error: %lu\n", GetLastError());
				VirtualFreeEx(hProcess, pRemoteBuf, 0, MEM_RELEASE);
				CloseHandle(hProcess);
				break;
			}

			WaitForSingleObject(hThread, INFINITE);
			printf(" [SUCCESS] DLL Injected successfully!\n");
			printf("           Check 'hook_captured_data.log' in the target directory for intercepted keys.\n");

			// Cleanup
			CloseHandle(hThread);
			VirtualFreeEx(hProcess, pRemoteBuf, 0, MEM_RELEASE);
			CloseHandle(hProcess);
		}
		break;
		case 99:
			exit(1);
			break;
		default:
			printf("Please select a menu option again.\n");
			break;
		}
	}
	// Release handles and snapshots
	CloseHandle(moduleSnapshotHandle);
	CloseHandle(targetProcessHandle);
	CloseHandle(snapshotHandle);

	//fclose(fp);
	return 0;
}

// 0
BOOL IsModuleLoaded(DWORD processID, LPCWSTR moduleName) {

	HMODULE hMods[1024] = { 0x00, };
	DWORD cbNeeded;
	char szModName[MAX_PATH] = { 0x00, };
	bool isEqual = true;
	char filename[256] = { 0x00, };
	char* ptr = NULL;


	HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, processID);
	if (hProcess == NULL) {
		return FALSE;
	}


	if (EnumProcessModules(hProcess, hMods, sizeof(hMods), &cbNeeded)) {
		for (unsigned int i = 0; i < (cbNeeded / sizeof(HMODULE)); i++) {

			if (GetModuleFileNameEx(hProcess, hMods[i], (LPWSTR)szModName, sizeof(szModName) / sizeof(wchar_t))) {
				ptr = strrchr(szModName, '\\');     // Search for the last '\' position in the string (path) from the end

				if (ptr == NULL)
					strcpy_s(filename, szModName);
				else
					strcpy_s(filename, ptr + 1); // Add +1 to pointer to extract filename only

				for (int i = 0; moduleName[i] != '\0'; i++) {
					if (_stricmp((const char*)PathFindFileName((LPCWSTR)filename), (const char*)moduleName) == 0) {
						isEqual = false;
						break;
					}
				}

				if (isEqual) {
					CloseHandle(hProcess);
					return TRUE;
				}
			}
		}
	}

	CloseHandle(hProcess);
	return FALSE;
}


ModuleInfo2 GetModuleInfo(const char* targetDll) {
	ModuleInfo2 result = { 0 };

	// Create a snapshot of the system
	HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	if (snapshot == INVALID_HANDLE_VALUE) {
		printf("Failed to create process snapshot.\n");
		return result;
	}

	// Set the size of the structure before using it
	PROCESSENTRY32 processEntry;
	processEntry.dwSize = sizeof(PROCESSENTRY32);

	// Iterate through the processes
	if (Process32First(snapshot, &processEntry)) {
		do {
			// Check if the process is running
			if (processEntry.th32ProcessID > 0) {
				// Open the process to get a module snapshot
				HANDLE processHandle = OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, FALSE, processEntry.th32ProcessID);
				if (processHandle != NULL) {
					// Create a module snapshot
					HANDLE moduleSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, processEntry.th32ProcessID);
					if (moduleSnapshot != INVALID_HANDLE_VALUE) {
						// Set the size of the structure before using it
						MODULEENTRY32 moduleEntry;
						moduleEntry.dwSize = sizeof(MODULEENTRY32);

						// Iterate through the modules in the process
						if (Module32First(moduleSnapshot, &moduleEntry)) {
							do {
								// Check if the module name matches the target DLL
								//if (_stricmp(moduleEntry.szModule, targetDll) == 0) {
								// Convert all characters to lowercase for comparison
								char lowercaseProcessName[MAX_PATH];
								for (int i = 0; i < MAX_PATH; i++) {
									lowercaseProcessName[i] = tolower(moduleEntry.szModule[i]);
								}
								if (_stricmp(lowercaseProcessName, targetDll) == 0) {
									result.processId = processEntry.th32ProcessID;
									result.moduleBase = moduleEntry.modBaseAddr;
									break;
								}
							} while (Module32Next(moduleSnapshot, &moduleEntry));
						}

						// Close the module snapshot handle
						CloseHandle(moduleSnapshot);
					}

					// Close the process handle
					CloseHandle(processHandle);
				}
			}
		} while (Process32Next(snapshot, &processEntry));
	}

	// Close the process snapshot handle
	CloseHandle(snapshot);

	return result;
}



// ============================================================
// Phase 2: Memory Snapshot Creation Function
// ============================================================
// ReadMemoryAndSaveToFile_case22 - Improved version
//
// Function: Save target process memory as a snapshot
//           Modified to read only data sections and stack/heap regions of the target DLL,
//           selectively capturing regions with high likelihood of containing cryptographic keys
//
// Parameters:
//   - processHandle: Target process handle
//   - filename: File name to save the snapshot
//   - dllName: Target DLL name for analysis
//   - processId: Target process ID
//
// Features:
//   - Suspends process for data consistency
//   - Excludes .text (code) section, captures data sections only
//   - File format: [Header][Size][Data] repeating structure
void ReadMemoryAndSaveToFile_case22(HANDLE processHandle, const char* filename, const char* dllName, DWORD processId) {
	MEMORY_BASIC_INFORMATION mbi;  // Struct to hold memory region information
	SIZE_T bytesRead;              // Actual bytes read
	BYTE* buffer = NULL;           // Buffer to store memory data
	FILE* file = NULL;             // Snapshot file handle

	// Open snapshot file (binary write mode)
	if (fopen_s(&file, filename, "wb") != 0) {
		fprintf(stderr, " [ERROR] Cannot open file %s for writing. Error: %d\n", filename, errno);
		return;
	}

	// [FIX] Acquire DLL base address before suspend (CreateToolhelp32Snapshot compatibility)
	LPVOID dllBaseAddress = GetDllBaseAddress(processId, dllName);
	if (dllBaseAddress == NULL) {
		// Continue even if DLL not found, to capture Private memory (heap/stack)
		fprintf(stderr, " [WARNING] DLL '%s' not found in process. Will capture Private memory only.\n", dllName);
	} else {
		printf(" [DEBUG] DLL base address: 0x%p\n", dllBaseAddress);
	}

	// ====================================================
	// [FIX] 1. Atomicity & Consistency
	// Suspend process for snapshot consistency
	// ====================================================
	// NtSuspendProcess/NtResumeProcess are undocumented APIs, loaded dynamically
	typedef LONG(NTAPI* NtSuspendProcess_t)(HANDLE ProcessHandle);
	typedef LONG(NTAPI* NtResumeProcess_t)(HANDLE ProcessHandle);

	NtSuspendProcess_t pfnSuspend = (NtSuspendProcess_t)GetProcAddress(GetModuleHandleA("ntdll"), "NtSuspendProcess");
	NtResumeProcess_t pfnResume = (NtResumeProcess_t)GetProcAddress(GetModuleHandleA("ntdll"), "NtResumeProcess");

	// [IMPROVED] Suspend process (prevent data tearing)
	// Retry logic added: max 3 attempts, proceed with warning on failure
	bool suspend_success = false;
	if (pfnSuspend) {
		for (int retry = 0; retry < 3; retry++) {
			LONG status = pfnSuspend(processHandle);
			if (status == 0) {
				suspend_success = true;
				break;
			}
			// Brief wait before retry
			if (retry < 2) {
				Sleep(50);
			}
		}
		if (!suspend_success) {
			fprintf(stderr, " [WARNING] NtSuspendProcess failed after 3 retries. Data consistency may be affected.\n");
			fprintf(stderr, "           Proceeding with snapshot capture anyway...\n");
		}
	} else {
		fprintf(stderr, " [WARNING] NtSuspendProcess not available. Data consistency may be affected.\n");
	}

	SIZE_T addr = 0;
	int regions_captured = 0;  // [FIX] Track number of captured regions
	SIZE_T total_bytes_captured = 0;  // [FIX] Total bytes captured
	int regions_scanned = 0;  // [DEBUG] Total regions scanned
	int private_regions_found = 0;  // [DEBUG] Number of Private memory regions

	// [DEBUG] Verify first VirtualQueryEx call
	SIZE_T queryResult = VirtualQueryEx(processHandle, (LPCVOID)addr, &mbi, sizeof(mbi));
	if (queryResult == 0) {
		DWORD error = GetLastError();
		fprintf(stderr, " [ERROR] VirtualQueryEx failed on first call. Error: %lu\n", error);
		fclose(file);
		if (pfnResume) pfnResume(processHandle);
		return;
	}

	// Start loop after processing first result
	do {
		regions_scanned++;
		bool isTargetRegion = false;
		char regionType[32] = "Unknown";

		// [IMPROVED] Target only readable memory regions (no execute permission)
		// Paper Section 3.3.1 requirement: exclude .text section, data sections only
		if (mbi.State == MEM_COMMIT && (mbi.Protect & PAGE_GUARD) == 0 &&
			mbi.Protect != PAGE_NOACCESS) {

			// Exclude sections with execute permission (code sections)
			if ((mbi.Protect & PAGE_EXECUTE) != 0 ||
				(mbi.Protect & PAGE_EXECUTE_READ) != 0 ||
				(mbi.Protect & PAGE_EXECUTE_READWRITE) != 0 ||
				(mbi.Protect & PAGE_EXECUTE_WRITECOPY) != 0) {
				// Skip executable code
				addr += mbi.RegionSize;
				continue;
			}

			// DLL data sections (.data, .rdata, etc.)
			if (mbi.Type == MEM_IMAGE && dllBaseAddress && (SIZE_T)mbi.BaseAddress >= (SIZE_T)dllBaseAddress) {
				// Writable sections (including dynamically allocated cryptographic data)
				if ((mbi.Protect & PAGE_READWRITE) != 0 || (mbi.Protect & PAGE_WRITECOPY) != 0) {
					isTargetRegion = true;
					strcpy_s(regionType, sizeof(regionType), "DLL Data Section");
				} else if ((mbi.Protect & PAGE_READONLY) != 0) {
					// Read-only sections (contains constant data, possible static keys)
					isTargetRegion = true;
					strcpy_s(regionType, sizeof(regionType), "DLL Read-Only Section");
				}
			}
			// [FIX] Other MEM_IMAGE regions (main EXE, other DLLs, etc.)
			// Keys may be stored in EXE global variables or other DLLs, so must capture
			else if (mbi.Type == MEM_IMAGE) {
				if ((mbi.Protect & PAGE_READWRITE) != 0 ||
				    (mbi.Protect & PAGE_WRITECOPY) != 0 ||
				    (mbi.Protect & PAGE_READONLY) != 0) {
					isTargetRegion = true;
					strcpy_s(regionType, sizeof(regionType), "Image (EXE/Other DLL)");
				}
			}
			// Private memory (heap and stack)
			else if (mbi.Type == MEM_PRIVATE) {
				private_regions_found++;  // [DEBUG] Private region count
				if ((mbi.Protect & PAGE_READWRITE) != 0 ||
				    (mbi.Protect & PAGE_READONLY) != 0 ||
				    (mbi.Protect & PAGE_WRITECOPY) != 0) {
					isTargetRegion = true;
					strcpy_s(regionType, sizeof(regionType), "Private (Stack/Heap)");
				}
			}
			// [FIX] MEM_MAPPED regions (memory-mapped files, shared memory, etc.)
			// Some programs may store keys/IVs in mapped memory
			else if (mbi.Type == MEM_MAPPED) {
				if ((mbi.Protect & PAGE_READWRITE) != 0 ||
				    (mbi.Protect & PAGE_READONLY) != 0 ||
				    (mbi.Protect & PAGE_WRITECOPY) != 0) {
					isTargetRegion = true;
					strcpy_s(regionType, sizeof(regionType), "Mapped Memory");
				}
			}
		}

		if (isTargetRegion) {
			// Save memory region header information
			MemoryRegionHeader header;
			header.baseAddress = (SIZE_T)mbi.BaseAddress;
			header.regionSize = mbi.RegionSize;
			strcpy_s(header.regionType, sizeof(header.regionType), regionType);
			header.protection = mbi.Protect;
			header.threadId = GetThreadIdForAddress(processHandle, (SIZE_T)mbi.BaseAddress);
			header.timestamp = (unsigned long)time(NULL);

			// Write header
			if (fwrite(&header, sizeof(MemoryRegionHeader), 1, file) != 1) {
				fprintf(stderr, " [ERROR] Failed to write header to file %s\n", filename);
				fclose(file);
				// [FIX] Must resume process on error - fix bug where process remains suspended
				if (pfnResume) pfnResume(processHandle);
				return;
			}

			// Dynamic buffer allocation
			buffer = (BYTE*)malloc(mbi.RegionSize);
			if (buffer == NULL) {
				fprintf(stderr, " [ERROR] Failed to allocate buffer for region size %zu\n", mbi.RegionSize);
				fclose(file);
				// [FIX] Must resume process on error
				if (pfnResume) pfnResume(processHandle);
				return;
			}

			// Read the entire region at once
			if (ReadProcessMemory(processHandle, mbi.BaseAddress, buffer, mbi.RegionSize, &bytesRead)) {
				// Write actual bytes read size
				if (fwrite(&bytesRead, sizeof(SIZE_T), 1, file) != 1) {
					fprintf(stderr, " [ERROR] Failed to write size to file\n");
					free(buffer);
					fclose(file);
					// [FIX] Must resume process on error
					if (pfnResume) pfnResume(processHandle);
					return;
				}
				// Write data
				if (fwrite(buffer, 1, bytesRead, file) != bytesRead) {
					fprintf(stderr, " [ERROR] Failed to write data to file %s\n", filename);
					free(buffer);
					fclose(file);
					// [FIX] Must resume process on error
					if (pfnResume) pfnResume(processHandle);
					return;
				}
				// [FIX] Increment capture success counter
				regions_captured++;
				total_bytes_captured += bytesRead;
			}
			else {
				// Record size as 0 on read failure
				SIZE_T zeroSize = 0;
				fwrite(&zeroSize, sizeof(SIZE_T), 1, file);
				DWORD error = GetLastError();
				fprintf(stderr, " [WARNING] ReadProcessMemory failed at 0x%08X. Error: %lu\n",
					(unsigned int)mbi.BaseAddress, error);
			}

			free(buffer);
			buffer = NULL;
		}

		addr += mbi.RegionSize;
	} while (VirtualQueryEx(processHandle, (LPCVOID)addr, &mbi, sizeof(mbi)) == sizeof(mbi));

	// [DEBUG] Output scan results
	printf(" [DEBUG] Memory scan complete: %d regions scanned, %d private regions found\n",
		regions_scanned, private_regions_found);

	// [FIX] Resume process after snapshot
	if (pfnResume) {
		pfnResume(processHandle);
	}

	fclose(file);

	// [FIX] Output capture results - warn if snapshot is empty
	if (regions_captured == 0) {
		fprintf(stderr, " [WARNING] No memory regions captured! Snapshot file may be empty.\n");
		fprintf(stderr, " [DEBUG] Hint: Check if process has accessible private memory regions.\n");
	} else {
		printf(" [DEBUG] Memory snapshot saved to %s (%d regions, %zu bytes)\n",
			filename, regions_captured, total_bytes_captured);
	}
}


// Improved comparison function
void CompareFilesAndLogDifferences_case22(HANDLE processHandle, const char* file1, const char* file2, const char* logFile) {
	FILE* f1 = NULL, * f2 = NULL, * log = NULL;

	// Open files
	if (fopen_s(&f1, file1, "rb") != 0 || fopen_s(&f2, file2, "rb") != 0 || fopen_s(&log, logFile, "w") != 0) {
		fprintf(stderr, " [ERROR] Failed to open files\n");
		if (f1) fclose(f1);
		if (f2) fclose(f2);
		if (log) fclose(log);
		return;
	}

	fprintf(log, "===== Memory Difference Analysis Report =====\n");
	fprintf(log, "Time: %s", __TIMESTAMP__);
	fprintf(log, "\nComparing: %s vs %s\n\n", file1, file2);

	// Initialize symbols
	SymInitialize(processHandle, NULL, TRUE);

	MemoryRegionHeader header1, header2;
	BYTE* buffer1 = NULL;
	BYTE* buffer2 = NULL;
	SIZE_T bytesRead1, bytesRead2;
	int regionCount = 0;
	int totalChanges = 0;

	while (1) {
		// Read headers
		if (fread(&header1, sizeof(MemoryRegionHeader), 1, f1) != 1) break;
		if (fread(&header2, sizeof(MemoryRegionHeader), 1, f2) != 1) break;

		// Read actual data sizes (must read before header validation to maintain file sync)
		if (fread(&bytesRead1, sizeof(SIZE_T), 1, f1) != 1) break;
		if (fread(&bytesRead2, sizeof(SIZE_T), 1, f2) != 1) break;

		// [FIX] Skip data on header mismatch to maintain file sync
		if (header1.baseAddress != header2.baseAddress) {
			fprintf(log, " [WARNING] Base address mismatch at region %d (0x%zX vs 0x%zX)\n",
				regionCount, header1.baseAddress, header2.baseAddress);
			// Skip data to synchronize file pointers
			if (bytesRead1 > 0) fseek(f1, (long)bytesRead1, SEEK_CUR);
			if (bytesRead2 > 0) fseek(f2, (long)bytesRead2, SEEK_CUR);
			regionCount++;
			continue;
		}

		regionCount++;

		// [FIX] Skip if no data - skip both sides
		if (bytesRead1 == 0 && bytesRead2 == 0) {
			continue;
		}
		// [FIX] Skip the other side's data when only one side has no data
		if (bytesRead1 == 0) {
			fseek(f2, (long)bytesRead2, SEEK_CUR);
			continue;
		}
		if (bytesRead2 == 0) {
			fseek(f1, (long)bytesRead1, SEEK_CUR);
			continue;
		}

		// Buffer allocation (safe memory management)
		buffer1 = (BYTE*)malloc(bytesRead1);
		buffer2 = (BYTE*)malloc(bytesRead2);
		if (!buffer1 || !buffer2) {
			SAFE_FREE(buffer1);
			SAFE_FREE(buffer2);
			// [FIX] Skip data on buffer allocation failure to maintain file sync
			fseek(f1, (long)bytesRead1, SEEK_CUR);
			fseek(f2, (long)bytesRead2, SEEK_CUR);
			continue;
		}

		// Read data
		size_t read1 = fread(buffer1, 1, bytesRead1, f1);
		size_t read2 = fread(buffer2, 1, bytesRead2, f2);
		if (read1 != bytesRead1 || read2 != bytesRead2) {
			// [FIX] Skip remaining data on read failure to maintain file sync
			if (read1 < bytesRead1) fseek(f1, (long)(bytesRead1 - read1), SEEK_CUR);
			if (read2 < bytesRead2) fseek(f2, (long)(bytesRead2 - read2), SEEK_CUR);
			SAFE_FREE(buffer1);
			SAFE_FREE(buffer2);
			continue;
		}

		// Data comparison
		SIZE_T minSize = min(bytesRead1, bytesRead2);
		MemoryChange change = { 0 };
		change.capacity = 1024;
		change.oldData = (unsigned char*)malloc(change.capacity);
		change.newData = (unsigned char*)malloc(change.capacity);

		// [FIX] Gap-tolerant merging: merge into one block if identical byte gap is ≤ GAP_TOLERANCE
		// Fixes issue where long data (certificates/RSA keys) gets split by a few identical bytes
		#define GAP_TOLERANCE 8  // Gaps of 8 bytes or less are merged into the same block
		SIZE_T gap_count = 0;    // Current count of consecutive identical bytes

		for (SIZE_T i = 0; i < minSize; i++) {
			if (buffer1[i] != buffer2[i]) {
				// Change start
				if (change.length == 0) {
					change.startAddress = header1.baseAddress + i;
					change.currentAddress = change.startAddress;
				}
				// Non-contiguous change
				else if (change.currentAddress + 1 != header1.baseAddress + i) {
					// If gap is within tolerance, merge including identical bytes in between
					SIZE_T actual_gap = (header1.baseAddress + i) - (change.currentAddress + 1);
					if (actual_gap <= GAP_TOLERANCE) {
						// Include gap bytes in the block
						SIZE_T gap_start_offset = (change.currentAddress + 1) - header1.baseAddress;
						for (SIZE_T g = gap_start_offset; g < i; g++) {
							if (change.length >= change.capacity) {
								change.capacity *= 2;
								change.oldData = (unsigned char*)realloc(change.oldData, change.capacity);
								change.newData = (unsigned char*)realloc(change.newData, change.capacity);
							}
							change.oldData[change.length] = buffer1[g];
							change.newData[change.length] = buffer2[g];
							change.length++;
						}
					}
					else {
						// If gap is large, log previous change and start new block
						if (change.length > 0) {
							LogMemoryChange(processHandle, log, &change, header1.regionType, NULL);
							totalChanges++;
						}
						change.startAddress = header1.baseAddress + i;
						change.length = 0;
					}
					change.currentAddress = header1.baseAddress + i;
				}

				// Reset gap counter
				gap_count = 0;

				// Expand buffer
				if (change.length >= change.capacity) {
					change.capacity *= 2;
					change.oldData = (unsigned char*)realloc(change.oldData, change.capacity);
					change.newData = (unsigned char*)realloc(change.newData, change.capacity);
				}

				// Store data
				change.oldData[change.length] = buffer1[i];
				change.newData[change.length] = buffer2[i];
				change.length++;
				change.currentAddress = header1.baseAddress + i;
			}
			else {
				gap_count++;
				// If change block is in progress and gap is within tolerance, don't break yet
				if (change.length > 0 && gap_count <= GAP_TOLERANCE) {
					// Hold without breaking (next changed byte may follow)
					continue;
				}
				// Log if gap exceeds tolerance or no change block exists
				if (change.length > 0) {
					LogMemoryChange(processHandle, log, &change, header1.regionType, NULL);
					totalChanges++;
					change.length = 0;
					gap_count = 0;
				}
			}
		}
		#undef GAP_TOLERANCE

		// Process final change
		if (change.length > 0) {
			LogMemoryChange(processHandle, log, &change, header1.regionType, NULL);
			totalChanges++;
		}

		// ============================================================
		// PSEUDOCODE: Static Key/IV Candidate Search (Paper Section 3.3)
		// ============================================================
		// Search for cryptographic key candidates in memory snapshot
		// that remain unchanged between two snapshots (static keys).
		//
		// ALGORITHM:
		//   KEY_LENGTHS ← {standard symmetric, EC, RSA, DH key sizes}
		//                  (DES=8, AES=16/24/32, EC P-256/384/521,
		//                   RSA-1024..8192, DH up to 24576 bits)
		//
		//   FOR EACH keyLen IN KEY_LENGTHS:
		//     FOR offset ← 0 TO (minSize - keyLen) STEP ALIGNMENT:
		//       candidate ← buffer2[offset .. offset+keyLen-1]
		//
		//       /* Character Analysis */
		//       nonZeroCount   ← COUNT non-zero bytes in candidate
		//       printableCount ← COUNT printable ASCII bytes
		//       hexCharCount   ← COUNT hex characters [0-9a-fA-F]
		//
		//       /* Entropy Analysis */
		//       entropy ← Shannon_Entropy(candidate)
		//
		//       /* Multi-path Key Candidate Detection */
		//       isCandidate ← false
		//       IF all printable AND all non-zero:
		//         isCandidate ← true (user-input key / passphrase)
		//       ELIF all hex characters:
		//         isCandidate ← true (hex-encoded IV/nonce)
		//       ELIF entropy > τ_entropy AND nonZero ≥ 90%:
		//         isCandidate ← true (binary crypto key)
		//
		//       /* Static Key Confirmation */
		//       IF isCandidate AND buffer1[offset..] == candidate:
		//         LOG as "Static Key/IV Candidate"
		//         (unchanged between snapshots → likely master key)
		// ============================================================
		/* Implementation omitted for security — see paper Section 3.3 */

		SAFE_FREE(change.oldData);
		SAFE_FREE(change.newData);
		SAFE_FREE(buffer1);
		SAFE_FREE(buffer2);
	}

	fprintf(log, "\n===== Summary =====\n");
	fprintf(log, "Total regions analyzed: %d\n", regionCount);
	fprintf(log, "Total memory changes detected: %d\n", totalChanges);
	fprintf(log, "==================\n");

	SymCleanup(processHandle);
	fclose(f1);
	fclose(f2);
	fclose(log);

	printf(" [DEBUG] Analysis complete. %d changes found in %d regions.\n", totalChanges, regionCount);
}


// [IMPROVED] Phase 5: Entropy threshold by confidence level (new)
// Paper Section 3.6.1 requirement: threshold calibration based on empirical data
double get_entropy_threshold_for_confidence(int confidence_level) {
	// ============================================================
	// PSEUDOCODE: Confidence-Level Entropy Threshold
	// ============================================================
	// Maps statistical confidence level to entropy threshold:
	//   confidence_level → τ_entropy
	// Based on empirical measurement of cryptographic data entropy
	// distributions (Paper Section 3.4.2, Table 3)
	//
	// Higher confidence → stricter threshold → fewer false positives
	// ============================================================
	/* Implementation omitted — see paper Section 3.4.2 */
	return 7.50;
}

// ============================================================
// Phase 5: Shannon Entropy Calculation Function
// ============================================================
// calculate_entropy - Improved version
//
// Function: Calculate Shannon entropy of data to measure randomness
//           Encrypted data exhibits high entropy values (7.8-8.0 bits/byte)
//
// Parameters:
//   - data: Data buffer to analyze
//   - length: Data length (bytes)
//
// Return value:
//   - Shannon entropy value (0.0 ~ 8.0 bits/byte)
//   - 0.0: Completely uniform data (e.g., all 0x00)
//   - 7.8-8.0: Highly random (cryptographic key or ciphertext)
//   - 4.0-6.0: Compressed data or plaintext
//
// Principle:
//   H(X) = -Σ p(x) * log2(p(x))
//   where p(x) is the occurrence probability of each byte value
double calculate_entropy(const unsigned char* data, size_t length) {
	// ============================================================
	// PSEUDOCODE: Shannon Entropy Calculation (Paper Equation 4)
	// ============================================================
	// INPUT:  data[0..length-1] — byte array to analyze
	// OUTPUT: H(X) ∈ [0.0, 8.0] bits/byte
	//
	// ALGORITHM:
	//   1. IF length == 0 THEN RETURN 0.0
	//   2. freq[256] ← COUNT occurrences of each byte value in data
	//   3. H ← 0
	//   4. FOR each byte value i in [0, 255]:
	//        IF freq[i] > 0:
	//          p(i) ← freq[i] / length
	//          H ← H - p(i) × log₂(p(i))
	//   5. RETURN H
	//
	// INTERPRETATION:
	//   H ≥ 7.81 → 99% confidence cryptographic randomness
	//   H ≥ 7.50 → 95% confidence
	//   H ∈ [4.0, 6.0] → compressed data or structured text
	//   H < 4.0  → plaintext or repetitive data
	// ============================================================
	/* Implementation omitted for security — see paper Section 3.4.1 */
	return 0.0;
}

// ============================================================
// Phase 5: Cryptographic Symbol Check Function
// ============================================================
// is_crypto_symbol - v4.1 improved version (false positives -10%)
//
// Function: Determine if a debug symbol name is a crypto-related function
//           Memory addresses near crypto functions have higher key probability
//
// Parameters:
//   - symbolName: Debug symbol name (e.g., "CryptEncrypt", "AES_encrypt")
//
// Return value:
//   - 0: Not crypto-related
//   - 1: Partial match (low confidence) - symbol name contains crypto keyword
//   - 2: Exact match (high confidence) - exactly matches known crypto API
//
// v4.1 improvements:
//   - Added word boundary check to reduce false positives by 10%
//   - e.g., "secure_malloc" no longer matches (previously matched due to "secure")
int is_crypto_symbol(const char* symbolName) {
	// ============================================================
	// PSEUDOCODE: Cryptographic Symbol Detection (Paper Signal 6)
	// ============================================================
	// INPUT:  symbolName — debug symbol name string
	// OUTPUT: 0 (no match), 1 (partial match), 2 (exact match)
	//
	// ALGORITHM:
	//   1. Define CRYPTO_API_LIST ← known crypto API names from:
	//        - Windows CryptoAPI (CryptEncrypt, CryptGenKey, ...)
	//        - OpenSSL (AES_encrypt, EVP_EncryptInit, ...)
	//        - BCrypt (BCryptEncrypt, BCryptGenRandom, ...)
	//        - mbedTLS (mbedtls_aes_init, ...)
	//        - Generic keywords (aes, rsa, encrypt, cipher, ...)
	//
	//   2. Phase 1 — Exact Match (high confidence):
	//      FOR each api IN CRYPTO_API_LIST:
	//        IF case_insensitive_equal(symbolName, api):
	//          RETURN 2
	//
	//   3. Phase 2 — Substring Match with Word Boundary (low confidence):
	//      FOR each api IN CRYPTO_API_LIST:
	//        pos ← find_substring(symbolName, api)
	//        IF pos exists AND word_boundary_valid(pos, api):
	//          RETURN 1
	//      (word_boundary: preceded by start/'_', followed by end/'_')
	//
	//   4. RETURN 0
	// ============================================================
	/* Implementation omitted for security — see paper Section 3.6.6 */
	return 0;
}

// Function to check if data is potentially cryptographic
bool is_potential_crypto_data(const unsigned char* data, size_t length) {
	// ============================================================
	// PSEUDOCODE: Potential Cryptographic Data Detection
	// ============================================================
	// INPUT:  data[0..length-1] — byte array
	// OUTPUT: true if data appears to be cryptographic material
	//
	// ALGORITHM:
	//   1. IF length < MIN_CRYPTO_LENGTH THEN RETURN false
	//
	//   2. SCAN data for structural anomalies:
	//      - Track consecutive_zeros, consecutive_ff
	//      - IF long runs of 0x00 or 0xFF detected (> length/4):
	//          RETURN false  (not crypto — padding or uninitialized)
	//
	//   3. COUNT unique byte values across data → unique_bytes
	//
	//   4. Multi-path detection (reduces false negatives):
	//      Path A: Strong randomness
	//        unique_bytes > α₁ × length AND unique_bytes > τ₁
	//      Path B: Moderate randomness (weak PRNG detection)
	//        unique_bytes > α₂ × length AND unique_bytes > τ₂
	//      Path C: Weak randomness (structural analysis)
	//        unique_bytes > τ₃ AND no long repetitive runs
	//
	//      (α₁ > α₂, τ₁ > τ₂ > τ₃ — thresholds from empirical data)
	//
	//   5. RETURN Path_A OR Path_B OR Path_C
	// ============================================================
	/* Implementation omitted for security — see paper Section 3.5 */
	return false;
}

// Improved log recording function (Phase 5 enhanced)
void LogMemoryChange(HANDLE processHandle, FILE* log, MemoryChange* change, const char* regionType, MemoryChangeMap* changeMap) {
	// ============================================================
	// PSEUDOCODE: Memory Change Analysis & Classification Logging
	// (Paper Section 3.6 — Phase 3-5 Integrated Analysis)
	// ============================================================
	// INPUT:
	//   processHandle — target process handle (for symbol resolution)
	//   log           — output log file handle
	//   change        — MemoryChange struct {startAddress, oldData, newData, length}
	//   regionType    — memory region type string
	//   changeMap     — Phase 4 change map (nullable, for pattern integration)
	//
	// ALGORITHM:
	//   1. VALIDATE: IF change is NULL or length == 0 THEN RETURN
	//
	//   2. SYMBOL RESOLUTION:
	//      symbolName ← ResolveDebugSymbol(processHandle, change.startAddress)
	//
	//   3. MULTI-SIGNAL COLLECTION (Paper 6 Signals):
	//      Signal 1: entropy      ← calculate_entropy(change.newData)
	//      Signal 2: chi_square   ← calculate_chi_square(change.newData)
	//      Signal 3: is_uniform   ← is_uniform_distribution(chi_square)
	//      Signal 4: isCrypto     ← is_crypto_symbol(symbolName)
	//      Signal 5: isPotential  ← is_potential_crypto_data(change.newData)
	//      Signal 6: threads[]    ← find_all_threads_for_address(startAddress)
	//
	//   4. PHASE 4-5 INTEGRATION (Paper Section 3.6.7):
	//      IF changeMap is provided:
	//        pattern ← GetPatternFromChangeMap(changeMap, startAddress)
	//      ELSE:
	//        pattern ← NULL
	//
	//   5. ML-BASED 5-CLASS CLASSIFICATION (Paper Algorithm 1):
	//      mlResult ← ClassifyMemoryBlock(newData, length, regionType, pattern)
	//      confidence ← mlResult.heuristic_score
	//      IF thread correlation found: confidence += THREAD_BONUS
	//      CLAMP confidence to [0, 100]
	//
	//   6. DATA TYPE DETERMINATION:
	//      Map mlResult.predicted_class × mlResult.confidence to label:
	//        KEY class      → "Highly Likely Key" / "Likely Key" / "Possible Key"
	//        IV class       → "Highly Likely IV" / "Likely IV" / "Possible IV"
	//        CIPHERTEXT     → "Cipher Text" / "Encrypted Data"
	//        PLAINTEXT      → "Plain Text"
	//        NON_CRYPTO     → "Non-Cryptographic"
	//
	//   7. LOG OUTPUT:
	//      Write address range, length, region, symbol info
	//      Write 5-class probabilities (KEY, IV, CIPHERTEXT, PLAINTEXT, NON_CRYPTO)
	//      Write 6 signals analysis (F1-F10)
	//      Write hex dump of old/new data (truncated for large blocks)
	//      Highlight high-confidence crypto material
	// ============================================================
	/* Implementation omitted for security — see paper Section 3.6 */
}




// Function to find the base address of a specific DLL in the target process
LPVOID GetDllBaseAddress(DWORD processId, const char* dllName) {
	LPVOID dllBaseAddress = NULL;
	HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, processId);
	if (hSnapshot != INVALID_HANDLE_VALUE) {
		MODULEENTRY32 me32;
		me32.dwSize = sizeof(MODULEENTRY32);
		if (Module32First(hSnapshot, &me32)) {
			do {
				char lowercaseProcessName[MAX_PATH] = { 0 };
				int i = 0;
				for (i = 0; me32.szModule[i] && i < MAX_PATH - 1; i++) {
					lowercaseProcessName[i] = tolower(me32.szModule[i]);
				}
				lowercaseProcessName[i] = '\0'; // Null-terminate

				if (_stricmp(lowercaseProcessName, dllName) == 0) {
					dllBaseAddress = me32.modBaseAddr;
					break;
				}
			} while (Module32Next(hSnapshot, &me32));
		}
		CloseHandle(hSnapshot);
	}
	return dllBaseAddress;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Util Funcs
// Function: Safely convert string to lowercase
void ConvertToLowerCase(char* dest, const wchar_t* src, size_t maxLen) {
	if (dest == NULL || src == NULL) {
		return;
	}

	size_t i;
	for (i = 0; i < maxLen - 1 && src[i] != '\0'; ++i) {
		dest[i] = tolower(src[i]);
	}
	dest[i] = '\0'; // Null-termination for safety
}


// Function to safely convert string
void SafeStringToLower(const TCHAR* input, char* output, size_t outputSize) {
	size_t i = 0;
	while (i < outputSize - 1 && input[i] != '\0') {
		output[i] = tolower(input[i]);
		i++;
	}
	output[i] = '\0';
}


// Function to check if standard cryptographic key length
bool is_standard_key_length(size_t length) {
	// ============================================================
	// PSEUDOCODE: Standard Cryptographic Key Length Check
	// (Paper Signal 3: F4)
	// ============================================================
	// INPUT:  length — data block size in bytes
	// OUTPUT: true if length matches known cryptographic key sizes
	//
	// ALGORITHM:
	//   STANDARD_KEY_LENGTHS ← set of known key sizes covering:
	//     - Symmetric ciphers: DES, AES-128/192/256, Blowfish, etc.
	//     - Stream ciphers: ChaCha20, Salsa20, RC4, etc.
	//     - HMAC keys: SHA-1, SHA-256, SHA-384, SHA-512
	//     - EC keys: P-224, P-256, P-384, P-521 (private, compressed, uncompressed)
	//     - RSA keys: 1024-bit through 8192-bit (N/8 bytes)
	//     - DH parameters: up to 24576-bit
	//
	//   RETURN length ∈ STANDARD_KEY_LENGTHS
	// ============================================================
	/* Implementation omitted for security — see paper Section 3.4.3 */
	return false;
}

// Function to check if standard IV length
bool is_standard_iv_length(size_t length) {
	// ============================================================
	// PSEUDOCODE: Standard IV/Nonce Length Check (Paper Signal 3: F5)
	// ============================================================
	// STANDARD_IV_LENGTHS ← {block cipher IVs, stream cipher nonces}
	//   (e.g., 8 bytes for DES-CBC, 12 for GCM/ChaCha20,
	//    16 for AES-CBC, 24 for XSalsa20/XChaCha20)
	// RETURN length ∈ STANDARD_IV_LENGTHS
	// ============================================================
	/* Implementation omitted for security — see paper Section 3.4.3 */
	return false;
}

// Function to get thread ID associated with a given address
DWORD GetThreadIdForAddress(HANDLE processHandle, SIZE_T address) {
	HANDLE threadSnap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (threadSnap == INVALID_HANDLE_VALUE) return 0;

	THREADENTRY32 te32;
	te32.dwSize = sizeof(THREADENTRY32);
	DWORD threadId = 0;
	DWORD processId = GetProcessId(processHandle);

	if (Thread32First(threadSnap, &te32)) {
		do {
			if (te32.th32OwnerProcessID == processId) {
				// Simply return the first thread ID (main thread)
				// For more precise thread identification,
				// separate logic to track threads per memory region is needed
				threadId = te32.th32ThreadID;
				break;
			}
		} while (Thread32Next(threadSnap, &te32));
	}
	CloseHandle(threadSnap);
	return threadId;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Phase 4 & Phase 5: New Function Implementations
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Phase 5: Chi-square test function (uniform distribution check)
double calculate_chi_square(const unsigned char* data, size_t length) {
	// ============================================================
	// PSEUDOCODE: Chi-Square Uniformity Test (Paper Equation 5)
	// ============================================================
	// INPUT:  data[0..length-1] — byte array
	// OUTPUT: χ² statistic value ∈ [0, ∞)
	//
	// ALGORITHM:
	//   1. IF length < 256 THEN RETURN 0 (insufficient data)
	//   2. freq[256] ← COUNT occurrences of each byte value
	//   3. expected ← length / 256
	//   4. χ² ← 0
	//   5. FOR i = 0 TO 255:
	//        χ² ← χ² + (freq[i] - expected)² / expected
	//   6. RETURN χ²
	//
	// Low χ² (< critical value) → uniform distribution → likely crypto
	// ============================================================
	/* Implementation omitted for security — see paper Section 3.4.2 */
	return 0;
}

// Phase 5: Uniform distribution check (based on chi-square value)
bool is_uniform_distribution(double chi_square) {
	// ============================================================
	// PSEUDOCODE: Uniform Distribution Check
	// ============================================================
	// Compare χ² against critical value (df=255, α=0.05)
	// RETURN χ² < χ²_critical
	// ============================================================
	/* Implementation omitted — see paper Section 3.4.2 */
	return false;
}

// Phase 5: Find all related threads (enhanced multi-thread tracking - improved version)
int find_all_threads_for_address(HANDLE processHandle, SIZE_T address, DWORD* threads_out, int max_threads) {
	HANDLE threadSnap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (threadSnap == INVALID_HANDLE_VALUE) return 0;

	THREADENTRY32 te32;
	te32.dwSize = sizeof(THREADENTRY32);
	DWORD processId = GetProcessId(processHandle);
	int thread_count = 0;

	if (Thread32First(threadSnap, &te32)) {
		do {
			if (te32.th32OwnerProcessID == processId && thread_count < max_threads) {
				// Open thread handle
				HANDLE hThread = OpenThread(THREAD_QUERY_INFORMATION | THREAD_GET_CONTEXT, FALSE, te32.th32ThreadID);
				if (hThread) {
					CONTEXT context;
					context.ContextFlags = CONTEXT_FULL;

					// Get thread context (for stack pointer check)
					if (GetThreadContext(hThread, &context)) {
#if defined(_ARM64_) || defined(_M_ARM64)
						SIZE_T sp = context.Sp;
#elif defined(_WIN64)
						SIZE_T sp = context.Rsp;
#else
						SIZE_T sp = context.Esp;
#endif
						// [IMPROVED] Determine stack range (more precise heuristic)
						// Paper Section 3.6.5 requirement: exact stack range via TIB query
						SIZE_T stack_low, stack_high;

						// Method 1: Basic heuristic (SP ± range)
						// Expanded range to include deep stack calls
						stack_low = (sp > 0x200000) ? sp - 0x200000 : 0;  // 2MB below
						stack_high = sp + 0x20000;  // 128KB above

						// Method 2: TIB-based precise range (future implementation)
						// TODO: Query exact StackLimit/StackBase via NtQueryInformationThread
						// Currently heuristic-based, so range is set wider

						if (address >= stack_low && address < stack_high) {
							threads_out[thread_count] = te32.th32ThreadID;
							thread_count++;
						}
					}
					CloseHandle(hThread);
				}
			}
		} while (Thread32Next(threadSnap, &te32));
	}

	CloseHandle(threadSnap);
	return thread_count;
}

// Phase 4: Initialize memory change map
MemoryChangeMap* InitializeChangeMap(int capacity) {
	MemoryChangeMap* map = (MemoryChangeMap*)malloc(sizeof(MemoryChangeMap));
	if (!map) return NULL;

	map->histories = (AddressChangeHistory*)calloc(capacity, sizeof(AddressChangeHistory));
	if (!map->histories) {
		free(map);
		return NULL;
	}

	map->history_count = 0;
	map->max_histories = capacity;
	return map;
}

// Phase 4: Free memory change map (improved version - dynamic allocation handling)
void FreeChangeMap(MemoryChangeMap* map) {
	if (!map) return;

	// Free dynamically allocated memory for each history
	if (map->histories) {
		for (int i = 0; i < map->history_count; i++) {
			SAFE_FREE(map->histories[i].values);  // Free snapshot value array for each address
		}
		SAFE_FREE(map->histories);
	}
	SAFE_FREE(map);
}

// Phase 4: Add or update address history (improved version - dynamic allocation)
void AddAddressHistory(MemoryChangeMap* map, SIZE_T address, unsigned char value, int snapshot_idx, const char* region_type, int* last_idx_hint) {
	if (!map) return;

	AddressChangeHistory* history = NULL;
	int found_idx = -1;

	// [OPTIMIZATION] Use hint if provided
	if (last_idx_hint && *last_idx_hint >= 0 && *last_idx_hint < map->history_count) {
		if (map->histories[*last_idx_hint].address == address) {
			history = &map->histories[*last_idx_hint];
			found_idx = *last_idx_hint;
		}
		// Hint failed, but maybe it's the next one? (Sequential access pattern)
		else if (*last_idx_hint + 1 < map->history_count && map->histories[*last_idx_hint + 1].address == address) {
			history = &map->histories[*last_idx_hint + 1];
			found_idx = *last_idx_hint + 1;
		}
	}

	// Linear Search if hint failed
	if (!history) {
		int start_search = (last_idx_hint && *last_idx_hint >= 0) ? *last_idx_hint : 0;
		if (start_search >= map->history_count) start_search = 0;

		for (int i = start_search; i < map->history_count; i++) {
			if (map->histories[i].address == address) {
				history = &map->histories[i];
				found_idx = i;
				break;
			}
		}
		// Wrap around if not found
		if (!history && start_search > 0) {
			for (int i = 0; i < start_search; i++) {
				if (map->histories[i].address == address) {
					history = &map->histories[i];
					found_idx = i;
					break;
				}
			}
		}
	}

	// Update hint
	if (last_idx_hint && found_idx != -1) {
		*last_idx_hint = found_idx;
	}

	// New address case
	if (!history) {
		// Capacity expansion needed
		if (map->history_count >= map->max_histories) {
			int new_capacity = map->max_histories * 2;
			AddressChangeHistory* new_histories = (AddressChangeHistory*)realloc(
				map->histories, new_capacity * sizeof(AddressChangeHistory));
			if (!new_histories) return;

			// Initialize newly allocated area
			memset(new_histories + map->max_histories, 0,
				map->max_histories * sizeof(AddressChangeHistory));

			map->histories = new_histories;
			map->max_histories = new_capacity;
		}

		history = &map->histories[map->history_count];
		history->address = address;
		history->snapshot_count = 0;
		history->total_changes = 0;
		history->change_frequency = 0.0;
		strncpy_s(history->region_type, sizeof(history->region_type), region_type, _TRUNCATE);

		// [IMPROVED] Create snapshot storage with dynamic allocation
		history->max_snapshot_capacity = DEFAULT_SNAPSHOTS;
		history->values = (unsigned char*)calloc(history->max_snapshot_capacity, sizeof(unsigned char));
		if (!history->values) {
			fprintf(stderr, " [ERROR] Failed to allocate snapshot values for address 0x%08X\n", (unsigned int)address);
			return;
		}

		if (last_idx_hint) *last_idx_hint = map->history_count; // Update hint to new item
		map->history_count++;
	}

	// Store value and detect changes (dynamic allocation version)
	// Expand snapshot buffer if needed
	if (snapshot_idx >= history->max_snapshot_capacity) {
		int new_capacity = history->max_snapshot_capacity * 2;
		if (new_capacity > MAX_SNAPSHOTS_CAPACITY) {
			new_capacity = MAX_SNAPSHOTS_CAPACITY;
		}

		if (snapshot_idx >= new_capacity) {
			fprintf(stderr, " [WARNING] Snapshot index %d exceeds max capacity %d\n",
				snapshot_idx, MAX_SNAPSHOTS_CAPACITY);
			return;
		}

		unsigned char* new_values = (unsigned char*)realloc(history->values, new_capacity);
		if (!new_values) {
			fprintf(stderr, " [ERROR] Failed to expand snapshot storage\n");
			return;
		}

		// Initialize newly allocated area
		memset(new_values + history->max_snapshot_capacity, 0,
			new_capacity - history->max_snapshot_capacity);

		history->values = new_values;
		history->max_snapshot_capacity = new_capacity;
	}

	// Store value
	unsigned char prev_value = (snapshot_idx > 0) ? history->values[snapshot_idx - 1] : 0;
	history->values[snapshot_idx] = value;

	if (snapshot_idx > 0 && prev_value != value) {
		history->total_changes++;
	}

	history->snapshot_count = snapshot_idx + 1;

	// Calculate change frequency
	if (history->snapshot_count > 1) {
		history->change_frequency = (double)history->total_changes / (double)(history->snapshot_count - 1);
	}
}

// [CRITICAL ISSUE #2] Phase 4-5 Integration: Query pattern info from ChangeMap (new)
// Paper Section 3.6.7 requirement: Pass Phase 4 pattern info to Phase 5 to improve classification accuracy
ChangePattern* GetPatternFromChangeMap(MemoryChangeMap* map, SIZE_T address) {
	if (!map || map->history_count == 0) {
		return NULL;  // No pattern info available
	}

	// Binary search (more efficient) - if sorted
	// Currently using linear search
	for (int i = 0; i < map->history_count; i++) {
		if (map->histories[i].address == address) {
			// Dynamically compute and return pattern
			// Note: Uses static variable within function for reusability
			static ChangePattern cached_pattern;
			cached_pattern = AnalyzeChangePattern(&map->histories[i]);
			return &cached_pattern;
		}
	}

	return NULL;  // Address not found
}

// Phase 4: Change pattern analysis
ChangePattern AnalyzeChangePattern(const AddressChangeHistory* history) {
	// ============================================================
	// PSEUDOCODE: Temporal Change Pattern Analysis (Paper Signal 5)
	// ============================================================
	// INPUT:  history — AddressChangeHistory with values[] across N snapshots
	// OUTPUT: ChangePattern {pattern_name, entropy, unique_values,
	//                        stability_score, change_frequency}
	//
	// ALGORITHM:
	//   1. IF history is NULL or snapshot_count == 0:
	//        RETURN pattern with name = "Unknown"
	//
	//   2. change_frequency ← history.change_frequency
	//        (pre-computed: total_changes / (snapshot_count - 1))
	//
	//   3. COMPUTE temporal entropy over snapshot values:
	//      freq[256] ← COUNT value occurrences across all snapshots
	//      entropy ← Shannon_Entropy(freq, snapshot_count)
	//
	//   4. unique_values ← COUNT distinct values in freq[]
	//
	//   5. stability_score ← (1.0 - change_frequency) × 100
	//
	//   6. CLASSIFY pattern by change_frequency thresholds:
	//        freq == 0.0       → "Static"           (master key candidate)
	//        freq < θ₁         → "Mostly Static"    (long-term key)
	//        freq < θ₂         → "Partially Changing"
	//        freq < θ₃         → "Frequently Changing" (session key)
	//        freq == 1.0       → "Always Changing"   (ephemeral key)
	//        otherwise         → "Mixed"
	//      (θ₁ < θ₂ < θ₃ — empirically determined thresholds)
	//
	//   7. RETURN pattern
	// ============================================================
	/* Implementation omitted for security — see paper Section 3.5.2 */
	ChangePattern pattern;
	memset(&pattern, 0, sizeof(ChangePattern));
	strcpy_s(pattern.pattern_name, sizeof(pattern.pattern_name), "Unknown");
	return pattern;
}

// Phase 4: Generate change log
void GenerateChangeLog(MemoryChangeMap* map, const char* output_file) {
	if (!map) return;

	FILE* log = NULL;
	if (fopen_s(&log, output_file, "w") != 0) {
		fprintf(stderr, " [ERROR] Cannot open log file %s\n", output_file);
		return;
	}

	fprintf(log, "====================================\n");
	fprintf(log, "Phase 4: Memory Change Tracking Report\n");
	fprintf(log, "====================================\n\n");
	fprintf(log, "Total addresses tracked: %d\n\n", map->history_count);

	// Analyze addresses with changes, or those without changes but high entropy (static key candidates)
	int changed_count = 0;
	for (int i = 0; i < map->history_count; i++) {
		const AddressChangeHistory* history = &map->histories[i];

		// Pre-analyze change pattern
		ChangePattern pattern = AnalyzeChangePattern(history);

		// Filter: ignore if no changes and low entropy (simple static data)
		// However, include as static key candidate if entropy >= 6.0
		if (history->total_changes == 0 && pattern.entropy < 6.0) continue;

		changed_count++;

		fprintf(log, "\n====== Address Change #%d ======\n", changed_count);
		fprintf(log, "Address: 0x%08X\n", (unsigned int)history->address);
		fprintf(log, "Region: %s\n", history->region_type);
		fprintf(log, "Total Changes: %d / %d snapshots\n",
			history->total_changes, history->snapshot_count - 1);
		fprintf(log, "Change Frequency: %.2f%%\n", history->change_frequency * 100.0);

		// Change pattern analysis (already performed)
		fprintf(log, "\nPattern Analysis:\n");
		fprintf(log, "  Type: %s\n", pattern.pattern_name);
		fprintf(log, "  Entropy: %.2f bits\n", pattern.entropy);
		fprintf(log, "  Unique Values: %d\n", pattern.unique_values);
		fprintf(log, "  Stability Score: %.1f/100\n", pattern.stability_score);

		// Value change history (display up to 20)
		fprintf(log, "\nValue History (first 20):\n");
		int max_display = (history->snapshot_count < 20) ? history->snapshot_count : 20;
		for (int j = 0; j < max_display; j++) {
			bool changed = (j > 0 && history->values[j] != history->values[j - 1]);
			fprintf(log, "  Snapshot %d: 0x%02X %s\n",
				j, history->values[j],
				changed ? "[CHANGED]" : "");
		}
		if (history->snapshot_count > 20) {
			fprintf(log, "  ... (%d more snapshots)\n", history->snapshot_count - 20);
		}
	}

	fprintf(log, "\n====================================\n");
	fprintf(log, "Summary:\n");
	fprintf(log, "  Total addresses with changes: %d\n", changed_count);
	fprintf(log, "  Total addresses tracked: %d\n", map->history_count);
	fprintf(log, "====================================\n");

	fclose(log);
	printf(" [DEBUG] Phase 4 change log written to %s\n", output_file);
}

// Phase 5: Classification score calculation (multi-signal integration)
int CalculateClassificationScore_Phase5(const unsigned char* data, size_t length,
	const char* region_type, const ChangePattern* pattern) {
	// ============================================================
	// PSEUDOCODE: Multi-Signal Classification Score (Paper Section 3.4)
	// ============================================================
	// INPUT:  data[], length, region_type, pattern (from Phase 4)
	// OUTPUT: score ∈ [0, 100] — cryptographic confidence score
	//
	// ALGORITHM:
	//   score ← 0
	//
	//   /* Signal 1: Shannon Entropy (max W₁ points) */
	//   entropy ← calculate_entropy(data, length)
	//   score += ENTROPY_SCORE_TABLE[entropy]
	//     // Multi-tier: ≥7.81→W₁, ≥7.5→W₁', ≥7.0→W₁'', ...
	//     // Thresholds aligned with statistical confidence levels
	//
	//   /* Signal 2: Chi-Square Uniformity (max W₂ points) */
	//   χ² ← calculate_chi_square(data, length)
	//   IF is_uniform(χ²):  score += W₂
	//   ELSE IF χ² < τ_mid: score += W₂'
	//   ELSE:               score += W₂''
	//
	//   /* Signal 3: Length Constraints (max W₃ points) */
	//   IF is_standard_key_length(length): score += W₃
	//   ELSE IF is_standard_iv_length(length): score += W₃'
	//   ELSE IF length ≥ 16: score += W₃''
	//
	//   /* Signal 4: Memory Region Type (max W₄ points) */
	//   score += REGION_WEIGHT_TABLE[region_type]
	//     // DLL Data Section > Stack/Heap > Other
	//
	//   /* Signal 5: Cross-Signal Synergy (max W₅ points) */
	//   IF entropy ≥ τ_high AND is_standard_key_length(length):
	//     score += W₅  // Strong key candidate bonus
	//
	//   /* Signal 6: Temporal Change Pattern (max W₆ points) */
	//   IF pattern is available:
	//     score += PATTERN_WEIGHT_TABLE[pattern.pattern_name]
	//       // Always Changing (session key) > Frequently Changing
	//       // > Mostly Static (master key) > Partially Changing
	//
	//   RETURN MIN(score, 100)
	//
	// NOTE: Weight values W₁..W₆ and thresholds τ are empirically
	//       calibrated — see paper Table 3 for exact values.
	// ============================================================
	/* Implementation omitted for security — see paper Section 3.4 */
	return 0;
}

// ============================================================
// Phase 5: ML-based Multi-Class Classification Function Implementation (Paper Section 3)
// ============================================================

// Convert class enum to string
const char* GetClassName(CryptoClassification cls) {
	switch (cls) {
		case CRYPTO_CLASS_KEY:        return "KEY";
		case CRYPTO_CLASS_IV:         return "IV";
		case CRYPTO_CLASS_CIPHERTEXT: return "CIPHERTEXT";
		case CRYPTO_CLASS_PLAINTEXT:  return "PLAINTEXT";
		case CRYPTO_CLASS_NON_CRYPTO: return "NON_CRYPTO";
		default:                      return "UNKNOWN";
	}
}

// Convert string region type to enum (Paper Signal 4: F6)
MemoryRegionType ParseRegionType(const char* region_type_str) {
	if (!region_type_str) return REGION_UNKNOWN;

	if (strcmp(region_type_str, "DLL Data Section") == 0 ||
	    strstr(region_type_str, ".data") != NULL ||
	    strstr(region_type_str, ".rdata") != NULL) {
		return REGION_DLL_DATA;
	}
	else if (strcmp(region_type_str, "Private (Stack/Heap)") == 0 ||
	         strstr(region_type_str, "Stack") != NULL ||
	         strstr(region_type_str, "Heap") != NULL) {
		return REGION_STACK_HEAP;
	}
	else if (strcmp(region_type_str, "Unknown") == 0) {
		return REGION_UNKNOWN;
	}
	return REGION_OTHER;
}

// Temporal pattern classification (Paper Signal 5: F8)
TemporalPatternType GetTemporalPatternType(const ChangePattern* pattern) {
	// ============================================================
	// PSEUDOCODE: Temporal Pattern Classification (Paper Equation F8)
	// ============================================================
	// INPUT:  pattern.change_frequency ∈ [0.0, 1.0]
	// OUTPUT: {STATIC, PARTIAL, FREQUENT, ALWAYS_CHANGING}
	//
	// Classify based on change_frequency thresholds:
	//   freq ≤ θ₁  → STATIC          (master key candidate)
	//   freq ≤ θ₂  → PARTIAL         (periodic key rotation)
	//   freq ≤ θ₃  → FREQUENT        (dynamic/session key)
	//   freq > θ₃  → ALWAYS_CHANGING (ephemeral key)
	//
	// θ₁ < θ₂ < θ₃ — thresholds calibrated to N snapshots
	// ============================================================
	/* Implementation omitted — see paper Section 3.5.2 */
	return PATTERN_STATIC;
}

// 10-Feature vector extraction (Paper Table 2)
// Paper Section 3.2: Extract 10 features from 6 signals
FeatureVector ExtractFeatureVector(const unsigned char* data, size_t length,
                                    const char* region_type, const ChangePattern* pattern) {
	// ============================================================
	// PSEUDOCODE: 10-Feature Vector Extraction (Paper Table 2)
	// ============================================================
	// INPUT:  data[], length, region_type, pattern
	// OUTPUT: FeatureVector with 10 features (F1-F10)
	//
	// FEATURE EXTRACTION from 6 Signals:
	//
	//   Signal 1 → F1:  Shannon Entropy
	//     F1 ← H(data) = -Σ p(i) log₂(p(i))          [0, 8] bits/byte
	//
	//   Signal 2 → F2:  Chi-Square Statistic
	//     F2 ← χ²(data) = Σ (O_i - E_i)² / E_i       [0, ∞)
	//
	//   Signal 3 → F3, F4, F5:  Length Constraints
	//     F3 ← length                                   [1, 1024] bytes
	//     F4 ← 1{length ∈ STANDARD_KEY_LENGTHS}        {0, 1}
	//     F5 ← 1{length ∈ STANDARD_IV_LENGTHS}         {0, 1}
	//
	//   Signal 4 → F6:  Memory Region Type
	//     F6 ← ParseRegionType(region_type)             [0, 3]
	//
	//   Signal 5 → F7, F8:  Temporal Change Pattern
	//     F7 ← change_frequency × N_snapshots           [0, N]
	//     F8 ← ClassifyTemporalPattern(pattern)         [0, 3]
	//          {STATIC, PARTIAL, FREQUENT, ALWAYS_CHANGING}
	//
	//   Signal 6 → F9, F10:  Cross-Signal Synergy
	//     F9  ← F1 × log₂(F3 + 1)                      (interaction term)
	//     F10 ← 1{F1 ≥ τ_entropy AND F4 == 1}          {0, 1}
	//          (high-confidence key indicator)
	//
	// RETURN feature_vector
	// ============================================================
	/* Implementation omitted for security — see paper Section 3.2 */
	FeatureVector fv;
	memset(&fv, 0, sizeof(FeatureVector));
	return fv;
}

// Hybrid ensemble classification (Paper Algorithm 1)
// Combines heuristic scoring with rule-based classification
// Future integration with ML models (XGBoost, Random Forest, Neural Network) possible
ClassificationResult ClassifyMemoryBlock(const unsigned char* data, size_t length,
                                          const char* region_type, const ChangePattern* pattern) {
	// ============================================================
	// PSEUDOCODE: Hybrid Ensemble Classification (Paper Algorithm 1)
	// ============================================================
	// INPUT:  data[], length, region_type, pattern
	// OUTPUT: ClassificationResult {predicted_class, confidence,
	//         heuristic_score, class_probabilities[5], class_name}
	//
	// Classes C = {KEY, IV, CIPHERTEXT, PLAINTEXT, NON_CRYPTO}
	//
	// ALGORITHM:
	//
	// Step 1: FEATURE EXTRACTION
	//   fv ← ExtractFeatureVector(data, length, region_type, pattern)
	//
	// Step 2: HEURISTIC SCORING
	//   H_score ← CalculateClassificationScore_Phase5(data, ...)
	//
	// Step 3: RULE-BASED PROBABILITY ESTIMATION
	//   Initialize P(c) = 0 for each class c ∈ C
	//
	//   3.1 Entropy-based:
	//     IF F1 ≥ τ_high:   boost P(KEY), P(IV), P(CIPHERTEXT)
	//     ELIF F1 ≥ τ_mid:  moderate boost to crypto classes
	//     ELIF F1 ≥ τ_low:  boost P(PLAINTEXT)
	//     ELSE:             strong boost P(PLAINTEXT), P(NON_CRYPTO)
	//
	//   3.2 Length-based:
	//     IF F4 (standard key length):   boost P(KEY)
	//     IF F5 (standard IV length):    boost P(IV)
	//     IF length ≥ 64 AND aligned:    boost P(CIPHERTEXT)
	//
	//   3.3 Distribution-based:
	//     IF uniform distribution (F2):  boost P(KEY), P(IV), P(CIPHER)
	//     ELSE:                          boost P(PLAINTEXT), P(NON_CRYPTO)
	//
	//   3.4 Synergy feature:
	//     IF F10 (high-confidence key):  boost P(KEY)
	//
	//   3.5 Temporal pattern:
	//     ALWAYS_CHANGING → boost P(KEY) (session key)
	//     STATIC + high entropy → boost P(KEY) (master key)
	//
	// Step 4: PROBABILITY NORMALIZATION
	//   P(c) ← P(c) / Σ P(c)  for all c ∈ C
	//
	// Step 5: ARGMAX CLASS SELECTION
	//   best_class ← argmax_c P(c)
	//   max_prob   ← max_c P(c)
	//
	// Step 6: HYBRID ENSEMBLE DECISION (key innovation)
	//   IF max_prob ≥ τ_high:
	//     // High confidence: trust probability-based classification
	//     predicted ← best_class, conf ← max_prob
	//
	//   ELIF max_prob ≥ τ_low:
	//     // Medium confidence: cross-validate with heuristic
	//     H_class ← MapHeuristicToClass(H_score, fv)
	//     IF best_class == H_class:
	//       predicted ← best_class, conf ← max_prob  (agreement)
	//     ELSE:
	//       predicted ← best_class, conf ← max_prob × γ (disagreement penalty)
	//
	//   ELSE:
	//     // Low confidence: fallback to heuristic rules
	//     predicted ← DeterministicClassify(H_score, fv)
	//     conf ← H_score / 100
	//
	// Step 7: SET class_name from predicted_class
	//
	// RETURN result
	//
	// NOTE: Thresholds τ_high, τ_low, γ and boost weights are
	//       empirically calibrated — see paper Table 4.
	// ============================================================
	/* Implementation omitted for security — see paper Algorithm 1 */
	ClassificationResult result;
	memset(&result, 0, sizeof(ClassificationResult));
	result.predicted_class = CRYPTO_CLASS_NON_CRYPTO;
	strncpy_s(result.class_name, sizeof(result.class_name), "NON_CRYPTO", _TRUNCATE);
	return result;
}