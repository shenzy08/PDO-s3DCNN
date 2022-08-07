import conf.global_settings as settings

class Settings:
    def __init__(self, settings):

        for attr in dir(settings):
            setattr(self, attr, getattr(settings, attr))

settings = Settings(settings)