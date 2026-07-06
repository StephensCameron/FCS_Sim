CC = gcc
CFLAGS = -Wall -Wextra -O2 -I$(HOME)/.local/include
LDFLAGS = -L$(HOME)/.local/lib -lraylib -lGL -lm -ldl -lpthread -lrt -lX11 -lXrandr -lXinerama -lXcursor -lXi

TARGET = diffusion_sim
SRCS = main.c coord.c diffusion_sim.c confocal_volume.c time_trace.c
OBJS = $(SRCS:.c=.o)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean
