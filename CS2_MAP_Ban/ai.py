import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 엑셀 파일에서 데이터 불러오기
file_path = 'Blitz_CSGO_UserDater.xlsx'
df = pd.read_excel(file_path)

# 맵 목록
maps = [
    'de_dust2',
    'de_inferno',
    'de_mirage',
    'de_nuke',
    'de_overpass',
    'de_vertigo'
]

# SteamID가 76561199262465443인 유저 데이터 선택
user_76561199262465443 = df[df['SteamID'] == '76561199262465443']

# 랜덤으로 9명의 데이터 선택 (SteamID가 76561199262465443인 유저를 제외)
selected_users = df[df['SteamID'] != '76561199262465443'].sample(n=9, random_state=1)  # random_state 설정 추가

# 팀 A에 SteamID가 76561199262465443인 유저와 나머지 4명 배치
team_A = pd.concat([user_76561199262465443, selected_users.sample(n=4, random_state=2)])

# 팀 B에 남은 5명 배치
team_B = selected_users.drop(team_A.index)

# 각 맵별 팀 A와 팀 B의 평균 승률 계산 (퍼센트로 표시)
team_A_map_win_rates = {}
team_B_map_win_rates = {}

for map_name in maps:
    team_A_map_win_rates[map_name] = team_A[map_name].mean()  # 퍼센트로 변환하지 않고 계산
    team_B_map_win_rates[map_name] = team_B[map_name].mean()  # 퍼센트로 변환하지 않고 계산

# AI 모델 학습을 위한 데이터 준비
X = pd.DataFrame({
    'Team': ['A', 'B'] * len(maps),
    'Map': maps * 2,
    'WinRate': [team_A_map_win_rates[map_name] for map_name in maps] +
               [team_B_map_win_rates[map_name] for map_name in maps]
})

y = [0, 1] * len(maps)  # A를 0, B를 1로 인코딩

# 데이터 분할 (학습용과 테스트용)
X_train, X_test, y_train, y_test = train_test_split(X[['WinRate']], y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 초기화 및 학습
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 팀 A의 맵별 예상 승률 예측 (퍼센트로 표시)
team_A_predicted_win_rates = {}
for map_name in maps:
    win_rate = team_A_map_win_rates[map_name]
    predicted_win_rate = model.predict([[win_rate]])[0] * 100.0  # 예상 승률을 퍼센트로 변환
    team_A_predicted_win_rates[map_name] = predicted_win_rate

# 추천 금지 맵 추출
recommended_bans = [map_name for map_name, win_rate in team_A_predicted_win_rates.items() if win_rate < 50.0]

# JSON 파일로 저장
import json

output_json_path = 'team_a_map_recommendations.json'
output_data = {
    'team_A_map_predictions': team_A_predicted_win_rates,
    'recommended_bans': recommended_bans
}

with open(output_json_path, 'w') as json_file:
    json.dump(output_data, json_file)

print(f"결과가 {output_json_path}에 저장되었습니다.")
