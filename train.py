import cPickle as pickle

def load(filename):
  return pickle.load(open(filename, 'rb'))
  
if __name__ == '__main__':
  audio = load('MUS/MAPS_MUS-alb_se3_AkPnBsdf.pkl')
