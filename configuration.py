from easydict import EasyDict as edict

# initialization
__C = edict()
cfg = __C

# postgresql
__C.user = "postgres"
__C.password = "postgres"
__C.dbname = "postgres"
__C.host = "mygridinstance.cm9xeut9zywf.ca-central-1.rds.amazonaws.com"
__C.port = 5432
