#pragma once
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
unsigned long long getmem() {
  task_t task = MACH_PORT_NULL;
  struct task_basic_info t_info;
  mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

  assert(KERN_SUCCESS == task_info(mach_task_self(), TASK_BASIC_INFO,
                                   (task_info_t)&t_info, &t_info_count));
  return t_info.resident_size / 1024;
}
#else
#include <cstring>
int getmem() {
  int peakRealMem;
  char buffer[1024] = "";
  FILE* file = fopen("/proc/self/status", "r");
  while (fscanf(file, " %1023s", buffer) == 1)
    if (strcmp(buffer, "VmHWM:") == 0) fscanf(file, " %d", &peakRealMem);
  fclose(file);
  return peakRealMem;
}
#endif
