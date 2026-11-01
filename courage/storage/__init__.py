"""存储层：紧凑产物（artifacts）与跨会话信念。

注意：不在包导入时触发 DB 初始化，避免副作用。
"""

from . import artifacts

__all__ = ["artifacts"]
