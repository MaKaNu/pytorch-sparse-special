class SizeValueError(ValueError):
    def __init__(self, obj):
        super().__init__(f"{type(obj)} is defined as 3D Matrix. Fix size or indices attribute!")
