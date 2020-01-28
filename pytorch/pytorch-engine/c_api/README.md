# Step to build this project
- Download the libtorch from [link](https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.4.0.zip)
and place the libtorch folder under `c_api`
- Created the `build` folder under c_api 
```
# Under c_api folder 
mkdir build && cd build
```
- Run
```
cmake -DCMAKE_PREFIX_PATH=liborch ..
cmake --build .
```
Then you should be able to run the gradle build in the pytorch-engine level.