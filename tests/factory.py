import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(
  __file__)+"/.."
)

from mlp.function import Factory, Type

class Test(unittest.TestCase):

  def test_A(self):
    name = None
    func = Factory() 
    for i in range(len(list(Type))):
      if i == 0: continue # 0 is unknown
      func.set(Type(i))
      valu = -1
      name = str(Type(i))
      what = func.evaluate(valu)
      print("%s %s" % (("Assert {}.evaluate(-1)".format(name.capitalize(), valu)).ljust(64, '.'), what))
 
if __name__ == '__main__':
  unittest.main(verbosity = 0)
