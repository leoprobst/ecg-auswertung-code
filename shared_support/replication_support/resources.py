"""Nur eigenen Prozess überwachen; Speicherknappheit wartet ohne Hauptlaufeingriff."""
import ctypes, time, threading, os, json
from pathlib import Path

class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ('tp','ap','tc','ac','tv','av','ae')]

def available():
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise RuntimeError('Speicherstatus nicht messbar')
    return {'ram_free_bytes':int(m.ap),'commit_free_bytes':int(m.ac)}

def guard():
    while True:
        m=available()
        if min(m.values())>=4*1024**3:return m
        print(json.dumps({'phase':'waiting_memory',**m}),flush=True);time.sleep(5)

class Budget:
    def __init__(self,out):self.out=Path(out);self.done=threading.Event();self.peak_rss=0;self.peak_commit=0
    def __enter__(self):
        import psutil
        p=psutil.Process()
        def watch():
            while not self.done.is_set():
                m=p.memory_info();self.peak_rss=max(self.peak_rss,m.rss,getattr(m,'peak_wset',0));self.peak_commit=max(self.peak_commit,getattr(m,'private',m.vms))
                if max(self.peak_rss,self.peak_commit)>1.5*1024**3:
                    self.out.mkdir(parents=True,exist_ok=True)
                    (self.out/'memory_stop.json').write_text(json.dumps(self.summary()),encoding='utf-8')
                    os._exit(75)
                self.done.wait(.1)
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start();return self
    def summary(self):return {'peak_rss_bytes':self.peak_rss,'peak_private_commit_bytes':self.peak_commit}
    def __exit__(self,*args):self.done.set();self.thread.join()
