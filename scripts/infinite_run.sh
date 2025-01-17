#!/bin/bash
SESSION_NAME="python_script_session"
LOG_FILE="logs/main_$(date +%Y_%m_%d).log"
PYTHON_SCRIPT="scripts/main.py"
INTERVAL=900  # 15 minutes in seconds
MAX_RETRIES=3
RETRY_DELAY=60  # 1 minute
LOG_TIMEOUT=60  # 10 minutes in seconds

echo "Current working directory: $(pwd)"

# Function to format duration in human readable format
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local remaining_seconds=$((seconds % 60))
    echo "${hours}h ${minutes}m ${remaining_seconds}s"
}

# Function to log messages
log_message() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$timestamp - $1" >> "$LOG_FILE"
}

# Function to check last log modification time
check_log_activity() {
    local current_time=$(date +%s)
    local last_modified=$(stat -f %m "$LOG_FILE" 2>/dev/null || stat -c %Y "$LOG_FILE")
    local time_difference=$((current_time - last_modified))
    
    if [ $time_difference -gt $LOG_TIMEOUT ]; then
        log_message "No log activity for ${time_difference} seconds (threshold: ${LOG_TIMEOUT} seconds). Initiating safe shutdown..."
        return 1
    fi
    return 0
}

# Function to kill existing tmux session if it exists
kill_existing_session() {
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        log_message "Killing existing tmux session: $SESSION_NAME"
        tmux kill-session -t "$SESSION_NAME"
    fi
}

# Function to create and manage tmux session
execute_in_tmux() {
    local retry_count=0
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        # Record start time
        local start_time=$(date +%s)
        
        # Kill any existing session
        kill_existing_session
        
        # Create a new tmux session
        tmux new-session -d -s "$SESSION_NAME" "python3 $PYTHON_SCRIPT"
        
        # Monitor the tmux session
        while tmux has-session -t "$SESSION_NAME" 2>/dev/null; do
            # Check log activity every 30 seconds
            sleep 30
            if ! check_log_activity; then
                # Kill the tmux session
                tmux kill-session -t "$SESSION_NAME"
                log_message "Tmux session terminated due to log inactivity"
                return 2  # Special return code for timeout
            fi
        done
        
        # Get tmux session exit code
        local exit_code=$(tmux display-message -p '#{pane_exit_code}' -t "$SESSION_NAME")
        
        # Calculate execution time
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local formatted_duration=$(format_duration $duration)
        
        # Log execution result
        if [ "$exit_code" = "0" ]; then
            log_message "Script executed successfully (Exit Code: $exit_code, Duration: $formatted_duration)"
            return 0
        else
            retry_count=$((retry_count + 1))
            log_message "Script failed with exit code $exit_code (Attempt $retry_count of $MAX_RETRIES, Duration: $formatted_duration)"
            
            if [ $retry_count -lt $MAX_RETRIES ]; then
                log_message "Retrying in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
            fi
        fi
    done
    
    log_message "Script failed after $MAX_RETRIES attempts"
    return 1
}

# Function to restart the script
restart_script() {
    log_message "Restarting script..."
    exec "$0" "$@"
}

# Create log file if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

# Main loop
log_message "Starting periodic tmux execution of $PYTHON_SCRIPT"
while true; do
    # Get start time of the entire cycle
    cycle_start_time=$(date +%s)
    
    # Execute Python script in tmux
    execute_in_tmux
    execution_status=$?
    
    # If timeout occurred, restart the script
    if [ $execution_status -eq 2 ]; then
        restart_script
    fi
    
    # Calculate how long the entire cycle took
    cycle_end_time=$(date +%s)
    cycle_duration=$((cycle_end_time - cycle_start_time))
    formatted_cycle_duration=$(format_duration $cycle_duration)
    
    # Calculate sleep time
    sleep_time=$((INTERVAL - cycle_duration))
    
    # Log and sleep if needed
    if [ $sleep_time -gt 0 ]; then
        formatted_sleep_time=$(format_duration $sleep_time)
        log_message "Cycle completed in $formatted_cycle_duration. Sleeping for $formatted_sleep_time until next execution"
        sleep $sleep_time
    else
        log_message "Warning: Cycle took $formatted_cycle_duration, which is longer than interval ($INTERVAL seconds)"
        # Optional: sleep for a minimum time to prevent continuous execution
        log_message "Waiting 60 seconds before next execution"
        sleep 60
    fi
done