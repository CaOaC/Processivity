# Processivity

The processivity model for simulate chromosome jet and mitotic chromosome folding

## ✨ Features

- Simulates the processive grappling motor
- Supports chromatin jets and mitotic chromosome folding
- Implements Condensin I / II motor activity models with residence
- Compatible with OVITO for visualization of `.dump` and `.data` trajectory files

## 📦 Dependencies



- [CUDA](https://developer.nvidia.com/cuda-downloads) ≥ 11.2
- [Make](https://www.gnu.org/software/make/) ≥ 4.2.1
- gcc (GCC) 9.3.1
- OS: CentOS Linux 7 (Core) 7.9.2009

## 🚀 Usage



Clone the repository and compile the code:

```bash
git clone git@github.com:CaOaC/Processivity.git

cd Processsivity/src

# Prepare the input and output folders (准备输入和输出文件夹)
mkdir Input Output

# Compile and run the program (构建并运行程序)
make -j8

./kickModel
```





## 📺 Demo



You can load the following files from the **`Output`** **Output** directory into `OVITO` to visualize the simulation trajectory.

- `.dump` file: trajectory animation
- `.data` file: topological structure


<h3>🎞️ movie </h3>


Visualize for the mitotic chromosome folding by processive grappling motors 


<div align="center">
  <img src="./Media/Compaction.gif" width="500" alt="Visualization of chromosome folding process">
</div>




Visualize for the steady states


<div align="center">
  <img src="./Media/Processivity.gif" width="500" alt="Visualization of chromosome folding process">
</div>



## 🧾 Information

Here we provide the codes for the manuscript "Theory of Chromosome Structural Dynamics by Processive Loop Extrusion". The provided files contain the codes that were developed based on the processive grappling motor model to study chromatin jets and mitotic chromosome folding.