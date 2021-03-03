#mkdir build && cd build
cd build
cmake ..
#cmake .. -DTFLITE_ENABLE_GPU=ON
cmake --build . -j 4
cd ..
