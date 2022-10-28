import redis


class CustomRedis:
    def __init__(self):
        self.db = 12
        self.host = 'localhost'
        self.port = 6379
        self.redis_instance = self.create_instance()

    def create_instance(self):
        return redis.StrictRedis(
            host=self.host,
            port=self.port,
            db=self.db
        )

    def get_all(self, pattern="*"):
        items = {}
        for key in self.redis_instance.keys(pattern):
            items[key.decode("UTF-8")] = self.redis_instance.get(key).decode("UTF-8")
        return items

    def get(self, needed_key):
        for key in self.redis_instance.keys("*"):
            current_key = key.decode("UTF-8")
            if current_key == needed_key:
                return json.loads(self.redis_instance.get(key).decode("UTF-8"))

    def set(self, key, value):
        return self.redis_instance.set(key, json.dumps(value))

    def delete(self, key):
        if key in self.get_all():
            self.redis_instance.delete(key)
            print(f'--> [{key}] successfully delete!')
        else:
            print(f'--> [{key}] is not exists!')