PACKAGE=mlp
VERSION=1.0.0
PYV=py3
SUFFIX='3'

all:
	python$(SUFFIX) ./setup.py sdist bdist_wheel

test:  test1 test2 test3 test4 test5

test1: 
	python$(SUFFIX) tests/factory.py

test2:
	python$(SUFFIX) tests/config.py

test3:
	python$(SUFFIX) tests/rpc.py

test4:
	python$(SUFFIX) tests/util.py

test5:
	python$(SUFFIX) tests/logger.py

test_mail:
	python$(SUFFIX) tests/mail.py
	

install: 
	pip$(SUFFIX) install -U dist/$(PACKAGE)-$(VERSION)-$(PYV)-none-any.whl

uninstall:
	pip$(SUFFIX) uninstall $(PACKAGE)

clean:
	rm -Rf build dist $(PACKAGE).egg-info
	find . -name "*.pyc" -delete
	find . -name "_version.py" -delete
	find . -name "__pycache__" -delete
