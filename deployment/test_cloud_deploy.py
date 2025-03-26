#!/usr/bin/env python3
"""
Unit tests for the cloud deployment script (cloud_deploy.py).
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
import tempfile
from io import StringIO

# Add the current directory to the path to import cloud_deploy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_deploy


class TestCloudDeploy(unittest.TestCase):
    """Test cases for the cloud deployment script."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, mode='w+')
        self.temp_config.write(json.dumps({
            "aws": {
                "region": "test-region",
                "environment": "test-env",
                "project_name": "test-project"
            },
            "azure": {
                "region": "test-region",
                "environment": "test-env",
                "project_name": "test-project"
            },
            "gcp": {
                "region": "test-region",
                "zone": "test-zone",
                "environment": "test-env",
                "project_name": "test-project"
            }
        }))
        self.temp_config.close()

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary config file
        os.unlink(self.temp_config.name)

    @patch('cloud_deploy.shutil.which')
    def test_check_prerequisites_all_installed(self, mock_which):
        """Test check_prerequisites when all tools are installed."""
        mock_which.return_value = '/usr/bin/terraform'  # Any non-None value
        self.assertTrue(cloud_deploy.check_prerequisites('aws'))
        # Verify which was called with all required tools
        self.assertEqual(mock_which.call_count, 4)  # terraform, docker, git, aws

    @patch('cloud_deploy.shutil.which')
    def test_check_prerequisites_missing_tools(self, mock_which):
        """Test check_prerequisites when some tools are missing."""
        # Make terraform missing
        mock_which.side_effect = lambda cmd: None if cmd == 'terraform' else '/usr/bin/'+cmd
        self.assertFalse(cloud_deploy.check_prerequisites('aws'))

    def test_load_config_default(self):
        """Test loading default config when no file is provided."""
        config = cloud_deploy.load_config(None)
        self.assertEqual(config, cloud_deploy.DEFAULT_CONFIG)

    def test_load_config_custom(self):
        """Test loading config from a file."""
        config = cloud_deploy.load_config(self.temp_config.name)
        self.assertEqual(config['aws']['region'], 'test-region')
        self.assertEqual(config['azure']['environment'], 'test-env')
        self.assertEqual(config['gcp']['project_name'], 'test-project')

    def test_load_config_invalid_file(self):
        """Test loading config with an invalid file."""
        config = cloud_deploy.load_config('nonexistent-file.json')
        self.assertEqual(config, cloud_deploy.DEFAULT_CONFIG)

    @patch('cloud_deploy.subprocess.run')
    def test_run_command_success(self, mock_run):
        """Test run_command with a successful command."""
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(cloud_deploy.run_command(['echo', 'test']))
        mock_run.assert_called_once()

    @patch('cloud_deploy.subprocess.run')
    def test_run_command_capture_output(self, mock_run):
        """Test run_command with output capture."""
        mock_process = MagicMock()
        mock_process.stdout = b'test output'
        mock_run.return_value = mock_process
        output = cloud_deploy.run_command(['echo', 'test'], capture_output=True)
        self.assertEqual(output, 'test output')
        mock_run.assert_called_once()

    @patch('cloud_deploy.subprocess.run')
    def test_run_command_failure(self, mock_run):
        """Test run_command with a failing command."""
        mock_run.side_effect = Exception('Command failed')
        self.assertFalse(cloud_deploy.run_command(['false']))
        mock_run.assert_called_once()

    @patch('cloud_deploy.run_command')
    def test_deploy_aws(self, mock_run_command):
        """Test AWS deployment."""
        mock_run_command.return_value = True
        args = MagicMock(
            provider='aws', 
            region='us-west-2', 
            environment='prod', 
            project_name='test-project',
            profile='test-profile',
            destroy=False,
            verbose=True,
            deployment_type='full'
        )
        config = {'aws': {'region': 'us-east-1', 'environment': 'dev', 'project_name': 'default-project'}}
        self.assertTrue(cloud_deploy.deploy_aws(args, config))
        mock_run_command.assert_called_once()
        # Verify the command contains the right arguments
        cmd = mock_run_command.call_args[0][0]
        self.assertIn('--region', cmd)
        self.assertIn('us-west-2', cmd)  # Should use args value, not config
        self.assertIn('--environment', cmd)
        self.assertIn('prod', cmd)
        self.assertIn('--profile', cmd)
        self.assertIn('test-profile', cmd)
        self.assertIn('--verbose', cmd)
        # Should not have skip flags for full deployment
        self.assertNotIn('--skip-infra', cmd)
        self.assertNotIn('--skip-backend', cmd)
        self.assertNotIn('--skip-frontend', cmd)

    @patch('cloud_deploy.run_command')
    def test_deploy_aws_infra_only(self, mock_run_command):
        """Test AWS infrastructure-only deployment."""
        mock_run_command.return_value = True
        args = MagicMock(
            provider='aws', 
            region='us-west-2', 
            environment='prod', 
            project_name='test-project',
            profile=None,
            destroy=False,
            verbose=False,
            deployment_type='infra'
        )
        config = {'aws': {'region': 'us-east-1', 'environment': 'dev', 'project_name': 'default-project'}}
        self.assertTrue(cloud_deploy.deploy_aws(args, config))
        # Verify infra-only has skip flags for backend and frontend
        cmd = mock_run_command.call_args[0][0]
        self.assertNotIn('--skip-infra', cmd)
        self.assertIn('--skip-backend', cmd)
        self.assertIn('--skip-frontend', cmd)

    @patch('cloud_deploy.deploy_aws')
    @patch('cloud_deploy.deploy_azure')
    @patch('cloud_deploy.deploy_gcp')
    @patch('cloud_deploy.check_prerequisites')
    @patch('cloud_deploy.load_config')
    def test_deploy_provider_selection(self, mock_load_config, mock_check_prerequisites, 
                                      mock_deploy_gcp, mock_deploy_azure, mock_deploy_aws):
        """Test provider selection in deploy function."""
        mock_check_prerequisites.return_value = True
        mock_load_config.return_value = cloud_deploy.DEFAULT_CONFIG
        
        # Test AWS
        args = MagicMock(provider='aws', config=None)
        mock_deploy_aws.return_value = True
        self.assertTrue(cloud_deploy.deploy(args))
        mock_deploy_aws.assert_called_once()
        mock_deploy_azure.assert_not_called()
        mock_deploy_gcp.assert_not_called()
        
        # Reset mocks
        mock_deploy_aws.reset_mock()
        mock_deploy_azure.reset_mock()
        mock_deploy_gcp.reset_mock()
        
        # Test Azure
        args = MagicMock(provider='azure', config=None)
        mock_deploy_azure.return_value = True
        self.assertTrue(cloud_deploy.deploy(args))
        mock_deploy_aws.assert_not_called()
        mock_deploy_azure.assert_called_once()
        mock_deploy_gcp.assert_not_called()
        
        # Reset mocks
        mock_deploy_aws.reset_mock()
        mock_deploy_azure.reset_mock()
        mock_deploy_gcp.reset_mock()
        
        # Test GCP
        args = MagicMock(provider='gcp', config=None)
        mock_deploy_gcp.return_value = True
        self.assertTrue(cloud_deploy.deploy(args))
        mock_deploy_aws.assert_not_called()
        mock_deploy_azure.assert_not_called()
        mock_deploy_gcp.assert_called_once()

    @patch('cloud_deploy.run_command')
    def test_setup_environment(self, mock_run_command):
        """Test setup environment function."""
        mock_run_command.return_value = True
        self.assertTrue(cloud_deploy.setup_environment())
        mock_run_command.assert_called_once()
        cmd = mock_run_command.call_args[0][0]
        self.assertEqual(len(cmd), 2)
        self.assertIn('deploy.sh', cmd[0])
        self.assertEqual('--setup', cmd[1])

    @patch('cloud_deploy.run_command')
    def test_get_deployment_status_aws(self, mock_run_command):
        """Test getting deployment status for AWS."""
        mock_run_command.return_value = '{"Stacks": [{"StackStatus": "CREATE_COMPLETE"}]}'
        args = MagicMock(
            provider='aws',
            region='us-west-2',
            environment='prod',
            project_name='test-project',
            profile='test-profile',
            config=None
        )
        status = cloud_deploy.get_deployment_status(args)
        self.assertIn('Stacks', status)
        mock_run_command.assert_called_once()
        cmd = mock_run_command.call_args[0][0]
        self.assertEqual(cmd[0], 'aws')
        self.assertIn('--region', cmd)
        self.assertIn('us-west-2', cmd)
        self.assertIn('--profile', cmd)
        self.assertIn('test-profile', cmd)

    @patch('cloud_deploy.run_command')
    def test_get_deployment_status_non_json(self, mock_run_command):
        """Test handling non-JSON output in get_deployment_status."""
        mock_run_command.return_value = 'This is not JSON'
        args = MagicMock(
            provider='aws',
            region='us-west-2',
            environment='prod',
            project_name='test-project',
            profile=None,
            config=None
        )
        status = cloud_deploy.get_deployment_status(args)
        self.assertIn('output', status)
        self.assertEqual(status['output'], 'This is not JSON')

    @patch('cloud_deploy.setup_environment')
    @patch('cloud_deploy.deploy')
    @patch('cloud_deploy.get_deployment_status')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_setup(self, mock_parse_args, mock_get_status, mock_deploy, mock_setup):
        """Test main function with setup command."""
        mock_parse_args.return_value = MagicMock(command='setup')
        mock_setup.return_value = True
        self.assertEqual(cloud_deploy.main(), 0)
        mock_setup.assert_called_once()
        mock_deploy.assert_not_called()
        mock_get_status.assert_not_called()

    @patch('cloud_deploy.setup_environment')
    @patch('cloud_deploy.deploy')
    @patch('cloud_deploy.get_deployment_status')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_deploy(self, mock_parse_args, mock_get_status, mock_deploy, mock_setup):
        """Test main function with deploy command."""
        mock_parse_args.return_value = MagicMock(command='deploy')
        mock_deploy.return_value = True
        self.assertEqual(cloud_deploy.main(), 0)
        mock_setup.assert_not_called()
        mock_deploy.assert_called_once()
        mock_get_status.assert_not_called()

    @patch('cloud_deploy.setup_environment')
    @patch('cloud_deploy.deploy')
    @patch('cloud_deploy.get_deployment_status')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_status(self, mock_parse_args, mock_get_status, mock_deploy, mock_setup):
        """Test main function with status command."""
        mock_parse_args.return_value = MagicMock(command='status')
        mock_get_status.return_value = {"status": "OK"}
        self.assertEqual(cloud_deploy.main(), 0)
        mock_setup.assert_not_called()
        mock_deploy.assert_not_called()
        mock_get_status.assert_called_once()

    @patch('cloud_deploy.setup_environment')
    @patch('cloud_deploy.deploy')
    @patch('cloud_deploy.get_deployment_status')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_no_command(self, mock_parse_args, mock_get_status, mock_deploy, mock_setup):
        """Test main function with no command."""
        mock_parse_args.return_value = MagicMock(command=None)
        self.assertEqual(cloud_deploy.main(), 1)
        mock_setup.assert_not_called()
        mock_deploy.assert_not_called()
        mock_get_status.assert_not_called()


if __name__ == '__main__':
    unittest.main() 