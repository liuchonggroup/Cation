"""
python v3.7.9
@Project: code
@File   : zzytool.py
@Author : Zhiyuan Zhang
@Date   : 2021/11/23
@Time   : 18:55
"""


class PBar(object):
    """    Custom Process Bar    """
    def __init__(self, data, doc=None, style=">", color="blue", brush_num=0.001):
        """"""
        self.idx = 0
        self.data = data
        self.doc = doc if doc else "Process"
        self.style = style
        self._color = self.color(color)
        self.len = len(self)
        self._brush_num = self.brush_num(brush_num)

    def __iter__(self):
        """"""
        return self

    def __next__(self):
        """"""
        if self.idx < self.len:
            if not self.idx % self._brush_num:
                finish = (51 * self.idx) // self.len
                finish_style = self.style * finish
                residue = "-" * (50 - finish)
                print(f"\r \033[1;{self._color}m {self.doc}: \033[0m", end="")
                print(f"\033[1;{self._color}m [{finish_style}{residue}] \033[0m", end="")
                print(f"\033[1;{self._color}m [{self.idx} / {self.len}] \033[0m", end="")

            idx = self.idx
            self.idx += 1
            return self.data[idx]
        else:
            self.idx = 0
            raise StopIteration

    def __len__(self):
        """"""
        return len(self.data)

    @staticmethod
    def color(c):
        """    Return Bar setting about color    """
        dict_color = {
            "white": 0,
            "black": 30,
            "red": 31,
            "green": 32,
            "yellow": 33,
            "blue": 34,
            "purple": 35,
            "indigo": 36,
            "grey": 37
        }
        return dict_color[c]

    def brush_num(self, brush_num):
        """"""
        brush_num = brush_num if isinstance(brush_num, int) else int(brush_num * self.len)
        if brush_num > 1:
            return brush_num
        else:
            return 1


def p_bar(idx, length, doc='Precess', brush=1, color=34, style=">"):
    """"""
    now_precess = 0
    if idx // brush > now_precess:
        now_precess = idx // brush
        finish = (51 * idx) // length
        finish_style = style * finish
        residue = "-" * (50 - finish)
        print(f"\r \033[1;{color}m {doc}: [{finish_style}{residue}] [{idx} / {length}]\033[0m", end="")

        if finish == 50:
            print("\r ", end="")
