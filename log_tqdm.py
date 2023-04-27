from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from subprocess import run
from tqdm import tqdm

FILE = 'model_RN_2023-04-23_17-32-43_780452.log'
LIMIT = 150
MESSAGE = "Fitting 3 folds for each of 1 candidates, totalling 3 fits"


def get_message_count():
    a = run(['grep', '-c', MESSAGE, FILE], capture_output=True)
    return int(a.stdout)


tq = tqdm(total=LIMIT, miniters=1)

last_count = get_message_count()

tq.update(last_count)
tq.refresh()


class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global last_count
        if event.src_path != FILE:
            return
        now_count = get_message_count()
        tq.update(now_count - last_count)
        tq.refresh()
        last_count = now_count


if __name__ == '__main__':
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()

    try:
        observer.join()
    except KeyboardInterrupt:
        observer.stop()
    finally:
        tq.close()
    observer.join()
