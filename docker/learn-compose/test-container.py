# test_content.py - Write this file directly
import sys
import json


def test_packages():
  results = {}

  packages = {
    "boto3": ["client", "resource"],
    "docker": ["from_env", "APIClient"],
    "getpass": ["getuser"],
    "glob": ["glob"],
    "gzip": ["compress", "decompress"],
    "hashlib": ["md5", "sha256"],
    "json": ["dumps", "loads"],
    "logging": ["getLogger", "StreamHandler"],
    "os": ["path", "environ", "getcwd"],
    "platform": ["system", "release", "python_version"],
    "psutil": ["cpu_count", "virtual_memory"],
    "shutil": ["copy", "rmtree"],
    "socket": ["gethostname", "socket"],
    "sqlite3": ["connect", "Row"],
    "subprocess": ["run", "PIPE"],
    "sys": ["version", "platform"],
    "tarfile": ["open", "TarFile"],
    "time": ["time", "sleep"],
    "zipfile": ["ZipFile", "ZIP_DEFLATED"],
  }

  for package_name, attributes in packages.items():
    try:
      package = __import__(package_name)
      attr_results = {}
      for attr in attributes:
        attr_results[attr] = hasattr(package, attr)

      # Test basic functionality
      working = True
      try:
        if package_name == "json":
          package.dumps({"test": "value"})
        elif package_name == "hashlib":
          package.md5(b"test").hexdigest()
        elif package_name == "platform":
          package.system()
        elif package_name == "socket":
          package.gethostname()
      except Exception as e:
        working = False

      results[package_name] = {
        "imported": True,
        "attributes": attr_results,
        "working": working,
      }

    except ImportError as e:
      results[package_name] = {"imported": False, "attributes": {}, "working": False}

  score = sum(result["working"] for result in results.values()) / len(results)
  non_working_packages_names = []

  for result_key, result in results.items():
    if not result["working"]:
      non_working_packages_names.append(result_key)

  print(f"score: {score}")
  print(f"non_working_packages_names: {non_working_packages_names}")


if __name__ == "__main__":
  test_packages()
