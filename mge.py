import os
import re
import subprocess
from pathlib import Path
import sys

def setup_git_repo(directory):
    """Git 저장소 초기화"""
    git_dir = os.path.join(directory, '.git')
    if not os.path.exists(git_dir):
        subprocess.run(['git', 'init'], cwd=directory, shell=True)
        
    # 기본 설정
    subprocess.run(['git', 'config', 'user.name', 'jhon-rjf'], cwd=directory, shell=True)
    subprocess.run(['git', 'config', 'user.email', 'ykjung0917@kakao.com'], cwd=directory, shell=True)

def get_version_files(directory, base_filename):
    """특정 파일의 버전들을 찾아서 정렬된 리스트로 반환"""
    files = []
    filename_without_ext, ext = os.path.splitext(base_filename)
    
    # 버전 패턴 매칭 (예: a.py, a_1.py, a_2.py)
    pattern = re.compile(f"^{re.escape(filename_without_ext)}(_\\d+)?{re.escape(ext)}$")
    
    try:
        for file in os.listdir(directory):
            match = pattern.match(file)
            if match:
                # 버전 번호 추출
                version_match = re.search(r'_(\d+)$', os.path.splitext(file)[0])
                version = int(version_match.group(1)) if version_match else 0
                files.append((version, file))
    except Exception as e:
        print(f"파일 검색 중 오류 발생: {e}")
        return []
    
    if not files:
        print(f"'{base_filename}' 관련 버전 파일을 찾을 수 없습니다.")
        return []
    
    return sorted(files, key=lambda x: x[0])

def migrate_to_git(directory, base_filename):
    """특정 파일의 버전들을 Git 커밋으로 변환"""
    try:
        setup_git_repo(directory)
        versions = get_version_files(directory, base_filename)
        
        if not versions:
            return
        
        print(f"\n'{base_filename}'의 버전 파일들을 처리합니다...")
        print(f"발견된 버전: {len(versions)}개")
        
        # 각 버전을 커밋으로 변환
        for version, filename in versions:
            try:
                # 파일 복사 (윈도우 인코딩 처리)
                file_path = os.path.join(directory, filename)
                target_path = os.path.join(directory, base_filename)
                
                try:
                    # UTF-8로 먼저 시도
                    with open(file_path, 'r', encoding='utf-8') as source:
                        content = source.read()
                except UnicodeDecodeError:
                    # UTF-8 실패시 cp949로 시도
                    with open(file_path, 'r', encoding='cp949') as source:
                        content = source.read()
                
                with open(target_path, 'w', encoding='utf-8') as target:
                    target.write(content)
                
                # Git 커밋
                subprocess.run(['git', 'add', base_filename], cwd=directory, shell=True)
                version_str = f"버전 {version}" if version > 0 else "초기 버전"
                commit_message = f"{base_filename}: {version_str}"
                subprocess.run(['git', 'commit', '-m', commit_message], cwd=directory, shell=True)
                
                print(f"커밋 완료: {commit_message}")
                
            except Exception as e:
                print(f"버전 {version} 처리 중 오류 발생: {e}")
                continue
        
        print(f"\n완료! Git 로그를 확인하려면 'git log'를 실행하세요.")
        
    except Exception as e:
        print(f"처리 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    print("버전 파일 Git 변환 도구 (윈도우용)")
    print("-" * 40)
    
    # 사용자로부터 파일명 입력받기
    base_filename = input("버전 관리할 파일명을 입력하세요 (예: a.py): ").strip()
    
    if not base_filename:
        print("오류: 파일명을 입력해주세요.")
    else:
        # 현재 디렉토리에서 실행
        current_dir = os.getcwd()
        migrate_to_git(current_dir, base_filename)