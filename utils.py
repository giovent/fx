def log(type, msg, during = 'execution'):
  print('{} message during {}: \n'.format(type, during) + msg)

def readConfigurationFile(self, filename, field):
  # TODO: read json configuration file
  log('ERROR', 'Couldn\'t find config file {path} of field {field}'.format(path=filename, field=field))
  return {}