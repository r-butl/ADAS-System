#ifndef READ_FRAME_HPP
#define READ_FRAME_HPP

#include "frame_buffer.hpp"

// Thread function for reading frames
void* frameReaderThread(void* arg);

struct FrameReaderArgs {
        FrameBuffer* frameBuffer;
        std::string source;
};
#endif // READ_FRAME_HPP
