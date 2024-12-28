if [ -d "cp/build" ]; then
    rm -rf cp/build
fi

mkdir cp/build
g++ cp/main.cpp cp/canvas/canvas.cpp -o cp/build/app
