if [ -d "build" ]; then
    rm -rf "build"
fi

mkdir "build"
g++ main.cpp vector/vector.cpp canvas/canvas.cpp -o build/app
