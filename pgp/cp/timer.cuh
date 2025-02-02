#ifndef _TIMER_H_
#define _TIMER_H_

#include <chrono>

namespace Timer {
    struct TTimer {
        std::chrono::_V2::system_clock::time_point _start;
        std::chrono::_V2::system_clock::time_point _end;
    };

    void Start(TTimer *timer) {
        timer->_start = std::chrono::high_resolution_clock::now();
    }

    void Stop(TTimer *timer) {
        timer->_end = std::chrono::high_resolution_clock::now();
    }

    long long GetTime(TTimer *timer) {
        std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(timer->_end - timer->_start);
        return duration.count();
    }
}

#endif // _TIMER_H_