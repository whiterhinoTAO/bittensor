
import asyncio

class AsyncTaskManager:
    def __init__(loop=None):
        self.task_map = {}

    def set_event_loop(loop=None):
        if 

    def set_event_loop(self, loop=None):
        if loop == None:
            loop = asyncio.get_event_loop()
        self.loop = loop
        return self.loop
         
    def async_run(self, job, loop=None): 
        if loop == None:
            loop = self.loop
        return self.loop.run_until_complete(job)


