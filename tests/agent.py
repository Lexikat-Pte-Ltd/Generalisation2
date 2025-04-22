import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from typing import List

from src.agent.agent import EnvAgent, StrategyAgent
from src.data import EnvironmentInfo
from src.genner import Genner
from docker import DockerClient


class TestEnvAgent(unittest.TestCase):
  def setUp(self):
    self.initial_basic_env_info = EnvironmentInfo(
      total_size=1000, used_size=500, available_size=500, use_percent=50.0
    )
    self.in_con_path = Path("/test/path")
    self.env_agent = EnvAgent(self.initial_basic_env_info, self.in_con_path)

  def test_initialization(self):
    self.assertEqual(self.env_agent.sp_env_info_getter_codes, [])
    self.assertEqual(self.env_agent.tagged_chat_history, [])
    self.assertEqual(self.env_agent.bs_env_info_history, [self.initial_basic_env_info])
    self.assertEqual(self.env_agent.sp_env_info_history, [])
    self.assertEqual(self.env_agent.in_con_path, self.in_con_path)

  def test_get_initial_tch(self):
    initial_tch = self.env_agent.get_initial_tch()
    self.assertIsInstance(initial_tch, list)
    self.assertTrue(len(initial_tch) > 0)
    for item in initial_tch:
      self.assertIsInstance(item, tuple)
      self.assertEqual(len(item), 2)
      self.assertIsInstance(item[0], dict)
      self.assertIsInstance(item[1], str)

  def test_chat_history_property(self):
    self.env_agent.tagged_chat_history = [
      ({"role": "system", "content": "System message"}, "system"),
      ({"role": "user", "content": "User message"}, "user"),
    ]
    chat_history = self.env_agent.chat_history
    self.assertEqual(len(chat_history), 2)
    self.assertEqual(chat_history[0]["role"], "system")
    self.assertEqual(chat_history[1]["role"], "user")

  @patch("src.agent.logger")
  def test_debug_log_tch(self, mock_logger):
    self.env_agent.tagged_chat_history = [
      ({"role": "system", "content": "System message"}, "system"),
    ]
    self.env_agent.debug_log_tch("Test context")
    mock_logger.debug.assert_called()

  @patch("src.agent.logger")
  def test_debug_log_sp_eih(self, mock_logger):
    self.env_agent.sp_env_info_history = [["Special info 1", "Special info 2"]]
    self.env_agent.debug_log_sp_eih("Test context")
    mock_logger.debug.assert_called()

  @patch("src.agent.logger")
  def test_debug_log_bs_eih(self, mock_logger):
    self.env_agent.debug_log_bs_eih("Test context")
    mock_logger.debug.assert_called()

  @patch("src.agent.run_code_in_con")
  @patch("src.agent.is_valid_code_compiler")
  @patch("src.agent.is_valid_code_ast")
  def test_gen_single_sp_egc(self, mock_ast, mock_compiler, mock_run_code):
    mock_ast.return_value = (True, None)
    mock_compiler.return_value = (True, None)
    mock_run_code.return_value = (0, "Test output", "Test code")

    mock_genner = Mock(spec=Genner)
    mock_genner.generate_code.return_value = ("test_code", "raw_response")

    mock_docker_client = Mock(spec=DockerClient)

    code, success, new_tch = self.env_agent.gen_single_sp_egc(
      count=0,
      genner=mock_genner,
      backup_genner=mock_genner,
      docker_client=mock_docker_client,
      testing_container_id="test_container",
      max_attempts=1,
    )

    self.assertEqual(code, "test_code")
    self.assertTrue(success)
    self.assertIsInstance(new_tch, list)

  def test_execute_sp_egc_s(self):
    mock_docker_client = Mock(spec=DockerClient)
    mock_docker_client.containers.get().exec_run.return_value = (0, b"Test output")

    self.env_agent.sp_env_info_getter_codes = ["print('Test')"]
    results = self.env_agent.execute_sp_egc_s(mock_docker_client, "test_container")

    self.assertEqual(results, ["Test output"])

  def test_append_new_code(self):
    initial_length = len(self.env_agent.sp_env_info_getter_codes)
    self.env_agent.append_new_code("new_code", "test_tag")
    self.assertEqual(len(self.env_agent.sp_env_info_getter_codes), initial_length + 1)
    self.assertEqual(self.env_agent.sp_env_info_getter_codes[-1], "new_code")

  @patch("src.agent.yaml.dump")
  @patch("builtins.open")
  def test_save_data(self, mock_open, mock_yaml_dump):
    self.env_agent.tagged_chat_history = [
      ({"role": "system", "content": "System message"}, "system"),
    ]
    self.env_agent.sp_env_info_getter_codes = ["test_code"]

    self.env_agent.save_data(Path("/test/save/path"))

    mock_open.assert_called_once()
    mock_yaml_dump.assert_called_once()


class TestStrategyAgent(unittest.TestCase):
  def setUp(self):
    self.init_bs_env_info = [
      EnvironmentInfo(
        total_size=1000, used_size=500, available_size=500, use_percent=50.0
      )
    ]
    self.init_sp_env_info = [["Special info 1", "Special info 2"]]
    self.prev_strats = ["Previous strategy 1", "Previous strategy 2"]
    self.in_con_path = Path("/test/path")
    self.common_agent = StrategyAgent(
      init_bs_env_info_history=self.init_bs_env_info,
      init_sp_env_info_history=self.init_sp_env_info,
      prev_strats=self.prev_strats,
      in_con_path=self.in_con_path,
    )

  def test_initialization(self):
    self.assertEqual(self.common_agent.tagged_chat_history, [])
    self.assertEqual(self.common_agent.strats, [])
    self.assertEqual(self.common_agent.in_con_path, self.in_con_path)
    self.assertEqual(self.common_agent.init_bs_env_info_history, self.init_bs_env_info)
    self.assertEqual(self.common_agent.init_sp_env_info_history, self.init_sp_env_info)
    self.assertEqual(self.common_agent.prev_strats, self.prev_strats)

  def test_get_initial_tch(self):
    initial_tch = self.common_agent.get_initial_tch()
    self.assertIsInstance(initial_tch, list)
    self.assertTrue(len(initial_tch) > 0)
    for item in initial_tch:
      self.assertIsInstance(item, tuple)
      self.assertEqual(len(item), 2)
      self.assertIsInstance(item[0], dict)
      self.assertIsInstance(item[1], str)

  def test_chat_history_property(self):
    self.common_agent.tagged_chat_history = [
      ({"role": "system", "content": "System message"}, "system"),
      ({"role": "user", "content": "User message"}, "user"),
    ]
    chat_history = self.common_agent.chat_history
    self.assertEqual(len(chat_history), 2)
    self.assertEqual(chat_history[0]["role"], "system")
    self.assertEqual(chat_history[1]["role"], "user")

  @patch("src.agent.logger")
  def test_debug_log(self, mock_logger):
    self.common_agent.tagged_chat_history = [
      ({"role": "system", "content": "System message"}, "system"),
    ]
    self.common_agent.debug_log()
    mock_logger.debug.assert_called_once()

  def test_gen_strats(self):
    mock_genner = Mock(spec=Genner)
    expected_strats = ["Strategy 1", "Strategy 2"]
    expected_raw = "Raw response"
    mock_genner.generate_list.return_value = (expected_strats, expected_raw)

    strats, raw = self.common_agent.gen_strats(mock_genner)

    self.assertEqual(strats, expected_strats)
    self.assertEqual(raw, expected_raw)
    mock_genner.generate_list.assert_called_once_with(
      self.common_agent.tagged_chat_history
    )

  def test_update_env_info_state(self):
    fresh_bs_eih = [
      EnvironmentInfo(
        total_size=2000, used_size=1000, available_size=1000, use_percent=50.0
      )
    ]
    fresh_sp_eih = [["New special info 1", "New special info 2"]]

    # Add some tagged messages that should be updated
    self.common_agent.tagged_chat_history = [
      ({"role": "system", "content": "Old basic env info"}, "get_basic_env_plist"),
      ({"role": "user", "content": "Old special env info"}, "get_special_env_plist"),
      ({"role": "assistant", "content": "Other message"}, "other_tag"),
    ]

    changes = self.common_agent.update_env_info_state(fresh_bs_eih, fresh_sp_eih)

    self.assertEqual(changes, 2)  # Should have updated 2 messages
    self.assertEqual(
      len(self.common_agent.tagged_chat_history), 3
    )  # Total length should remain same

  @patch("src.agent.yaml.dump")
  @patch("builtins.open")
  def test_save_data(self, mock_open, mock_yaml_dump):
    self.common_agent.tagged_chat_history = [
      ({"role": "system", "content": "System message"}, "system"),
    ]

    self.common_agent.save_data(
      space_freed=100.0, strat="Test strategy", folder=Path("/test/save/path")
    )

    mock_open.assert_called_once()
    mock_yaml_dump.assert_called_once()


if __name__ == "__main__":
  unittest.main()
