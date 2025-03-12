__all__ = ["Media", "File", "Image", "Video"]


class Media:
    pass


class File(Media):
    def __init__(self, path: str) -> None:
        self.path = path


class Image(File):
    pass


class Video(File):
    def __init__(self, path: str, start_timestamp: int = -1, end_timestamp: int = -1) -> None:
        super().__init__(path)
        # You may choose to only view a segment of the video
        self.start_timestamp = start_timestamp # in seconds
        self.end_timestamp = end_timestamp # in seconds


