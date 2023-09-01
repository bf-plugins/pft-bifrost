import config
import redis
r = redis.Redis(config.redis_db_ip)

print('PKT_START %s' % r.get('snap:utc_start'))
