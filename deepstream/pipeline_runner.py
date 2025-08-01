#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import os
import signal
import sys
from pathlib import Path


class PipelineRunner(Node):
    def __init__(self):
        super().__init__('pipeline_runner')
        
        # Dictionary to keep track of running pipelines {pipeline_name: process_object}
        self.running_pipelines = {}
        
        # Path to pipeline scripts directory
        self.pipelines_dir = Path(__file__).parent / 'pipelines'
        
        # Create subscriber for pipeline commands
        self.command_subscriber = self.create_subscription(
            String,
            '/cortex_ai/pipeline_commands',
            self.command_callback,
            10
        )
        
        self.get_logger().info('Pipeline Runner initialized')
        self.get_logger().info(f'Pipeline scripts directory: {self.pipelines_dir}')
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Periodically check for finished child processes
        self.process_check_timer = self.create_timer(10.0, self.check_pipeline_processes)
    
    def command_callback(self, msg):
        """Handle incoming pipeline commands"""
        command = msg.data.strip()
        self.get_logger().info(f'Received command: {command}')

        if command == "kill":
            self.get_logger().info("Kill command received. Exiting now.")
            self.cleanup_all_pipelines()
            self.destroy_node()
            os._exit(0)

        try:
            if ':' not in command:
                self.get_logger().error(f'Invalid command format: {command}. Expected format: start:<pipeline_name> or stop:<pipeline_name>')
                return
            
            action, pipeline_name = command.split(':', 1)
            
            if action == 'start':
                self.start_pipeline(pipeline_name)
            elif action == 'stop':
                self.stop_pipeline(pipeline_name)
            else:
                self.get_logger().error(f'Unknown action: {action}. Valid actions are: start, stop')
                
        except Exception as e:
            self.get_logger().error(f'Error processing command "{command}": {str(e)}')
    
    def start_pipeline(self, pipeline_name):
        """Start a pipeline as a separate process"""
        if pipeline_name in self.running_pipelines:
            self.get_logger().warning(f'Pipeline "{pipeline_name}" is already running')
            return
        
        # Check if pipeline script exists
        pipeline_script = self.pipelines_dir / f'{pipeline_name}.py'
        if not pipeline_script.exists():
            self.get_logger().error(f'Pipeline script not found: {pipeline_script}')
            return
        
        try:
            # Start the pipeline as a separate process.
            # By not redirecting stdout/stderr to a PIPE, we avoid the child process
            # hanging when the pipe buffer fills up. The child's output will go to
            # the same console as this runner script.
            process = subprocess.Popen([
                sys.executable, str(pipeline_script)
            ])
            
            self.running_pipelines[pipeline_name] = process
            self.get_logger().info(f'Started pipeline "{pipeline_name}" with PID: {process.pid}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to start pipeline "{pipeline_name}": {str(e)}')
    
    def stop_pipeline(self, pipeline_name):
        """Stop a running pipeline"""
        if pipeline_name not in self.running_pipelines:
            self.get_logger().warning(f'Pipeline "{pipeline_name}" is not running')
            return
        
        try:
            process = self.running_pipelines[pipeline_name]
            
            # Check if process is still running
            if process.poll() is None:
                # Gracefully terminate the process
                process.terminate()
                
                # Wait for process to terminate (with timeout)
                try:
                    process.wait(timeout=5)
                    self.get_logger().info(f'Pipeline "{pipeline_name}" terminated gracefully')
                except subprocess.TimeoutExpired:
                    # Force kill if not terminated within timeout
                    process.kill()
                    process.wait()
                    self.get_logger().warning(f'Pipeline "{pipeline_name}" was force killed')
            else:
                self.get_logger().info(f'Pipeline "{pipeline_name}" was already terminated')
            
            # Remove from running pipelines
            del self.running_pipelines[pipeline_name]
            
        except Exception as e:
            self.get_logger().error(f'Error stopping pipeline "{pipeline_name}": {str(e)}')
    
    def cleanup_all_pipelines(self):
        """Stop all running pipelines"""
        self.get_logger().info('Stopping all running pipelines...')
        
        pipeline_names = list(self.running_pipelines.keys())
        for pipeline_name in pipeline_names:
            self.stop_pipeline(pipeline_name)
    
    def check_pipeline_processes(self):
        """Remove pipelines from tracking if their process has exited."""
        to_remove = []
        for name, proc in self.running_pipelines.items():
            if proc.poll() is not None:
                self.get_logger().info(f'Pipeline "{name}" has finished on its own (exit code {proc.returncode}). Removing from running list.')
                to_remove.append(name)
        for name in to_remove:
            del self.running_pipelines[name]
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.get_logger().info(f'Received signal {signum}, shutting down...')
        self.cleanup_all_pipelines()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    pipeline_runner = PipelineRunner()
    rclpy.spin(pipeline_runner)

if __name__ == '__main__':
    main()