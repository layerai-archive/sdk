set -eu

API_KEY=$1
RED='\033[0;31m'
NC='\033[0m'

echo "$API_KEY" > .test-token

start_time=$(date +%s%3N)
set +e
make e2e-test
exit_code=$?
set -e
end_time=$(date +%s%3N)

echo ""
echo -e "${RED}Logs${NC}: https://app.datadoghq.eu/logs?from_ts=$start_time&to_ts=$end_time&index=%2A&live=false&query=stack%3Aapp"
echo -e "${RED}Untrusted logs${NC}: https://app.datadoghq.eu/logs?from_ts=$start_time&to_ts=$end_time&index=%2A&live=false&query=stack%3Auntrusted-prod%20kube_namespace%3Auntrusted-app"
echo -e "${RED}APM traces${NC}: https://app.datadoghq.eu/apm/traces?start=$start_time&end=$end_time&index=%2A&live=false&paused=true&query=stack%3Aapp"
echo -e "${RED}Test traces${NC}: https://app.datadoghq.eu/ci/test-runs?query=%40ci.pipeline.id%3A$GITHUB_RUN_ID%20&index=citest&start=$start_time&end=$end_time&paused=true"
echo ""

exit $exit_code
