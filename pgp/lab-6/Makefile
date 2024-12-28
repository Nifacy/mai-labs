# Compiler and flags
CXX = g++
CXXFLAGS = -lGL -lGLU -lglut -Wall

# Source and target
SRC = main.cpp
TARGET = app

# Compile the program
compile:
	$(CXX) $(SRC) -o $(TARGET) $(CXXFLAGS)

# Compile and run the program
run: compile
	./$(TARGET)

# Clean up the build
clean:
	rm -f $(TARGET)
