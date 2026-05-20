## The AI Performance Engineering Meetup

- This is a recap of an online [meetup](https://www.meetup.com/ai-performance-engineering/events/311657001/?eventOrigin=home_page_event_from_group_you_are_in)
- This meetup has been running for a while. There are many previous talks on the [AI Performance Engineering Meetup - YouTube Channel](https://www.youtube.com/@AIPerformanceEngineering)
- The meetup is organized by [Chris Fregly](https://www.linkedin.com/in/cfregly/) and others.
- Chris Fregly is
  - the author of the book [AI Performance Engineering](https://www.amazon.com/AI-Performance-Engineering-Model-Optimization/dp/1098122180) and
  - runs a [blog](https://cfregly.medium.com/) on Medium.
  - Maintains a [GitHub repository](https://github.com/cfregly/ai-performance-engineering) with code examples from the book and code and slides from the talks. Though on first inspection the repo only covers a few of the talks

## Diving deep into NVIDIA Nsight Systems GPU profiling tools for PyTorch LLM and computer vision workloads

- [Chaim Rand](https://www.linkedin.com/in/chaim-rand-7692721/?originalSubdomain=il)

- In this talk, Chaim Rand revisits the [NVIDIA Nsight](https://developer.nvidia.com/nsight-systems) profiling tools to augment the PyTorch Profiler.

- This talk is based on Chaim’s recent articles series “A Deep Dive Using NVIDIA Nsight™ Systems Profiler”

  - part 1: [Optimizing Data Transfer in AI/ML Workloads](https://chaimrand.medium.com/optimizing-data-transfer-in-ai-ml-workloads-60df62fe1278) and
  - part 2: [Optimizing Data Transfer in Batched AI/ML Inference Workloads](https://chaimrand.medium.com/optimizing-data-transfer-in-batched-ai-ml-inference-workloads-a9f4165208b8).

These cover two issues that are familiar to anyone who took a course where one builds and trains deep learning models

1.  Moving the data to and from the GPU is much slower than the speed it is processed within the GPU. This is so slow that if the number of Matrix operations isn’t very big it may faster to run the model on the CPU than on the GPU! Anyone familiar with Flash attention and the mamba architecture will realize that this is not a trivial issue as even within the GPU there are multiple levels of memory with different speeds and latencies.
2.  Early books on deep learning discussed how batching can speed up training and that it is the main innovation for Stochastic Gradient Descent. It was recognized early that large batches with samples that are dissimilar lead to faster convergence. On the other hand it is also a challenge to fit batches of data with differing sizes into the memory of the GPU. Padding is one solution but it leads to waste of memory and compute. Dynamic batching is another solution but it leads to overheads in managing the batches. In inference workloads this is particularly challenging as the requests come in at different times and with different sizes.

These two earlier talks discuss how to use the PyTorch Profiler to deal with these two issues.

there are a couple of previous videos by Chaim Rand on AI Performance Engineering Meetup

# An error occurred.

Unable to execute JavaScript.

Modern AI model training often relies on powerful and expensive machinery like GPUs. Performance optimization is a key tool in accelerating model convergence and reducing development costs. Fortunately, you do not need to be a CUDA expert in order to introduce meaningful improvements to your training efficiency. In this talk, we will motivate the inclusion of performance analysis and optimization in your AI model development and demonstrate a few simple tricks and techniques for accelerating your model. c.f. [slides](https://github.com/cfregly/ai-performance-engineering/blob/main/resources/PyTorch_Model_Optimization.pdf)

# An error occurred.

Unable to execute JavaScript.

c.f. [slides](https://github.com/cfregly/ai-performance-engineering/blob/main/resources/PyTorch_Model_Optimization_Data_Loader.pdf)

Since life is short I’ve put the videos for the first two parts in the margin.

Chaim is a great content creator and speaker. However with so many articles and videos it is hard to find them all. I’ve linked a few resources below. Also some point are more fundamental and get repeated in different talks.

- AI/ML developers must take responsibility for the runtime performance of their workloads
- You don’t need to be a CUDA expert to see results.

#### Optimization Methodology

- Objective:- Maximize throughput (samples per second)
- Use performance profilers to measure resource utilization and identify bottlenecks
- Integrate into model development lifecycle
  1.  Adding reporting to the model’s code so that the profilers can report performance metrics as part of the model’s training and inference processes. This is often just a matter of adding one or two lines of code to the model’s training and inference scripts.
  2.  Using the profiler’s visualization tools to analyze the performance data and identify bottlenecks. There is two common undercurrent in software engineering that
      - “if you can’t measure it you can’t improve it”.
      - “avoid early optimization” Data scientist often end up waiting while the model trains and then only look at the final results. But the lesson here is that perhaps we should be measuring performance as part of the model development lifecycle prior to running any big training runs as this could save us time and money. The culprit is that most notebooks used to teach us about deep learning/ML/AI do not cover this side of the craft focusing just on a single metric (e.g. precision)

### A Suggested recipe

1.  **Profile** (use a profiler e.g. [PyTorch Profiler]() or [NVIDIA Nsight]() to identify bottlenecks)
2.  **Optimize** i.e. Fix bottlenecks using optimization techniques such as caching, data loading, memory management, kernel fusion, mixed precision training, or many of the other techniques discussed in the talks and articles.
3.  **Repeat** until the GPU is well utilized (i.e. no red blocks or gaps appear in the profiler) and or other performance goals are met (e.g. latency, throughput, cost etc).

There are a number of steps in a typical AI/ML workload and a principled approach to optimization is to profile each step and find the bottlenecks. In the previous talks on the pytorch profiler Chaim laid out his optimization methodology.

perhaps most useful is his youtube channel with previous talks in this series:

says devs/ and data scientists should be resource that they are utilizing thier GPUS (i.e. profiling)

------------------------------------------------------------------------

This talk shows shots of the Nvidas Nsight GPU profiler.

We start with images of the profile in a bad way - red blocked or idle blocks!

do 1-3 intervention and remove the red in the profiler getting 2-4x speedup.

the profiling relies on using NVTX block annotations

### Resources

- [AI Performance Engineering](https://www.youtube.com/@AIPerformanceEngineering)

- [medium blog](https://chaimrand.medium.com/)

- [Optimizing Data Transfer in AI/ML Workloads](https://chaimrand.medium.com/optimizing-data-transfer-in-ai-ml-workloads-60df62fe1278) A Deep Dive Using NVIDIA Nsight™ Systems Profiler — Part 1

- [Optimizing Data Transfer in Batched AI/ML Inference Workloads](https://chaimrand.medium.com/optimizing-data-transfer-in-batched-ai-ml-inference-workloads-a9f4165208b8) A Deep Dive Using NVIDIA Nsight™ Systems Profiler — Part 2

- [slides and code](https://github.com/cfregly/ai-performance-engineering/)

- [Likwid](https://hpc-wiki.info/hpc/Likwid)

- Series on **PyTorch Model Performance Analysis and Optimization**

  1.  [How to Use PyTorch Profiler and TensorBoard to Accelerate Training and Reduce Cost](https://medium.com/@chaimrand/pytorch-model-performance-analysis-and-optimization-10c3c5822869)
  2.  [How to Identify and Reduce CPU Computation In Your Training Step with PyTorch Profiler and TensorBoard](https://medium.com/data-science/pytorch-model-performance-analysis-and-optimization-part-2-3bc241be91)
  3.  [How to reduce “Cuda Memcpy Async” events and why you should beware of boolean mask operations](https://medium.com/data-science/pytorch-model-performance-analysis-and-optimization-part-3-1c5876d78fe2)
  4.  [Solving Bottlenecks on the Data Input Pipeline with PyTorch Profiler and TensorBoard](https://medium.com/data-science/solving-bottlenecks-on-the-data-input-pipeline-with-pytorch-profiler-and-tensorboard-5dced134dbe9)
  5.  [How to Optimize Your DL Data-Input Pipeline with a Custom PyTorch Operator](https://medium.com/data-science/how-to-optimize-your-dl-data-input-pipeline-with-a-custom-pytorch-operator-7f8ea2da5206)
  6.  [PyTorch Model Performance Analysis and Optimization](https://medium.com/data-science/pytorch-model-performance-analysis-and-optimization-part-6-b87412a0371b)
  7.  [Efficient Metric Collection in PyTorch: Avoiding the Performance Pitfalls of TorchMetrics](https://chaimrand.medium.com/efficient-metric-collection-in-pytorch-avoiding-the-performance-pitfalls-of-torchmetrics-0dea81413681)
  8.  [A Caching Strategy for Identifying Bottlenecks on the Data Input Pipeline](https://chaimrand.medium.com/a-caching-strategy-for-identifying-bottlenecks-on-the-data-input-pipeline-8e52060b402f)
  9.  [Pipelining AI/ML Training Workloads With CUDA Streams](https://chaimrand.medium.com/pipelining-ai-ml-training-workloads-with-cuda-streams-bf5746449409)
  10. [The Role of NUMA Awareness](https://chaimrand.medium.com/the-crucial-role-of-numa-awareness-in-high-performance-deep-learning-99ae3e8eb49a)
  11. [Overcoming the Hidden Performance Traps of Variable-Shaped Tensors: Efficient Data Sampling in PyTorch](https://chaimrand.medium.com/overcoming-the-hidden-performance-traps-of-variable-shaped-tensors-efficient-data-sampling-in-9f4f4f29355e)

### Reflection & Some questions

- The long series covers the pytorch optimizer + [tensorboard](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) tool for visualization. It looks like tensorboard has been abandoned by google or at least deprecated in favor of other tools. Are there any other tools that one should be using today?
  - [Weights and Biases](https://wandb.ai/site) as an alternative.
  - [Perfetto](https://perfetto.dev/) - an open-source project for performance instrumentation and tracing of Android, Linux, Chrome, and other platforms.
  - Chrome via *chrome://tracing* c.f. [How to use the Chrome UI to analyze Pytorch Profiler traces?](https://www.youtube.com/watch?v=AhIOohJYSrw) c.f. Chrome Tracing ui:

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2026,
  author = {Bochman, Oren},
  title = {Nvidia {Nsight} {GPU} Profiling},
  date = {2026-01-19},
  url = {https://orenbochman.github.io/posts/2026/01-20-nvidia-Nsight-GPU-profiling/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2026. “Nvidia Nsight GPU Profiling.” January 19. <https://orenbochman.github.io/posts/2026/01-20-nvidia-Nsight-GPU-profiling/>.
