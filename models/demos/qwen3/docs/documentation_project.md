The purpose of this project is to create a series of documents summarizing and analyzing **Tenstorrent Hardware** and the **TT-Metal framework**.

---

# Important Instructions for the LLM  
- Create a separate document for each **Section** below.  
- Organize the **questions and answers** clearly by topic.  
- Merge similar questions into one where appropriate.  
- Address **only one topic at a time**; do not attempt to answer everything in one pass.  
- Provide **answers based on verified source code**, not assumptions.  
- Examine and reference **actual implementation code**, not just simple examples.  
- Include **example source code** in each document, and where relevant, **attach links to actual implementation files** showing the related code.

---

# Tenstorrent Hardware  

## Chip  
- How are harvested cores represented in **tt-metal**?  
	- How can the 2D torus of Tensix cores (e.g., 8×8 grid) be accessed conveniently?  
	- How to control harvested cores, DRAM cores, and Ethernet (ETH) cores together?  
	- How are **logical view** and **physical view** defined and managed? Are there separate APIs for each?  

## Tensix Core  
  - 2× NoC, each 64 B/clk for data movement  
  - 5× RISC-V cores  
  - 1.5 MB local SRAM accessible by all RISC-V cores in a Tensix Core  
  - Matrix engine, vector engine  

- In **TT-Metal**, what does “page-based access” to off-chip memory mean?  
- What is the difference between **interleaved** and **consecutive** allocation?

---

# Host Programming  

- **Kernel compilation:** `CreateKernel(coordinate, src.cpp, functionality, RISC-V core)`  
	- Does the `functionality` parameter limit which APIs in `src.cpp` can be used?  
	- Does the `functionality` parameter also restrict which RISC-V cores can execute the kernel?  
- How to handle **tile layout data**?  
- Can different Tensix cores execute **different kernels** in one operation?  
	- Example: In an 8×8 Tensix core grid, can 4×8 cores run Kernel A and the other 4×8 run Kernel B?  
- Summary of **C/C++ APIs**:  
	- Are APIs like `EnqueueWriteBuffer`, `EnqueueProgram` still in use? In what form?  
- How to use **TTNN Tensor** in both C++ host code and kernels?  
- How can each kernel be provided with **device information**? Can a kernel identify which device number it runs on?

---

# Kernel Programming  

- How is **local SRAM** allocated?  
- What **tt-metal API** reads data from DRAM into local SRAM?  
- How to use the **vector engine** and **matrix engine**?  
- How to read/write **tile layout data** between DRAM and SRAM?  
- For RISC-V 0 and RISC-V 4, can each use only one NoC, or can both use any NoC?  
	- Can two cores share the same NoC?  
	- Can one core use both NoCs?  
- **Intra-core synchronization** among RISC-V cores within a Tensix core:  
	- What is the total size of the circular buffer?  
	- Can a single kernel use multiple circular buffers?  
	- How to allocate and use a circular buffer?  
	- How to **share** a circular buffer between cores?  
	- How does one core signal that data is ready, and how do others wait for it?  
	- What are **PACK** and **UNPACK**? What APIs are provided for them?  
	- What other signaling methods exist between RISC-V cores?  
- What distinguishes **high-level kernel APIs** from **low-level kernel APIs**?  
- How to program the **Ethernet core**?  
- How to embed **device information** in each kernel? Can the kernel identify its device index?

---

# Communication between Tensix Cores  

- **Inter-core synchronization**:  
	- How do Tensix cores communicate with each other?  
	- How does one core write data to another core?  
	- How does one core read data from another core?  
	- Is a **semaphore** used for synchronization? If so, how is it created and used?  
- What is **data multicast**? What APIs are available for multicast, and how are they used? Include real examples.  
- How can a Tensix core **read/write data to DRAM** (DRAM core)?  
- When Tensix cores exchange data via SRAM, does that SRAM refer to the **1.5 MB local scratchpad** or the **32 KB shared memory** inside the math unit?  
- What are the **differences between NoC0 and NoC1** on the chip?  
- How to specify which **NoC** to use for data transmission?
