import sys
import re

from loguru import logger


class BufferToLogger:
    def __init__(self, level, regex, real_stdout) -> None:
        self.level = level
        if isinstance(regex, str):
            self.regex = re.compile(regex)
        elif isinstance(regex, re.Pattern):
            self.regex = regex
        else:
            raise ValueError(f'Tipo inválido para o parâmetro `regex` {type(regex)}')
        self.real_stdout = real_stdout
    
    def write(self, raw_message):
        message = raw_message.strip()
        if not message or not self.regex.match(message):
            print(raw_message, file=self.real_stdout, end='')
            return
        depth = 2
        frame = sys._getframe(depth)
        while frame and frame.f_code.co_filename == __file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth).log(self.level, message)
    
    def flush(self):
        return