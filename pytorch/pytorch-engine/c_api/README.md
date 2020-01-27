1. Download the libtorch from [link](https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.4.0.zip)
2. Created the `build` folder under c_api 
```
# Under c_api folder 
mkdir build && cd build
```
3. Run
```
cmake -DCMAKE_PREFIX_PATH=path/to/your/liborch ..
cmake --build .
```