CC = g++
CFLAGS = -Wall -Wextra -std=c++17 -I /usr/local/include/opencv4 -O0 -pthread
LDFLAGS = -lopencv_imgcodecs -lopencv_core -lopencv_highgui -lopencv_videoio 
TARGET = lab5
SRC = lab5.cpp

$(TARGET) : $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)