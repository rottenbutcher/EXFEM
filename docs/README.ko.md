# gfem 사용 안내 (한국어)

gfem은 JAX 기반의 GPU 가속 비선형 유한요소 해석 오픈소스입니다.
CAD 가져오기·모델링, 메싱, 해석, 결과 확인을 ANSYS Mechanical과 비슷한 화면에서 할 수 있습니다.

## 설치와 실행

```bash
pip install -e ".[all]"          # CPU용 (jax, scipy, gmsh, pyamg, ezdxf)
pip install "jax[cuda12]"        # NVIDIA GPU (코드 수정 없음)
gfem gui                         # 브라우저에서 http://127.0.0.1:8765 열림
gfem gui --host 0.0.0.0          # GPU 서버에서 실행하고 노트북 브라우저로 접속
gfem run job.inp                 # Abaqus 입력 파일 바로 해석 (.bdf, .cdb도 가능)
gfem convert job.inp job.bdf     # Abaqus ↔ Nastran 변환
gfem mesh part.step -o part.inp  # STEP 파일을 메싱해 해석 파일로 저장
```

## 화면 구성 (ANSYS Mechanical과 같은 흐름)

| 영역 | 내용 |
|---|---|
| 리본 | Home / Geometry / Mesh / Environment / Solution / Results / Display |
| Outline 트리 | Geometry(피처 트리, 바디) → Materials → Connections → Named Selections → Mesh → 해석(설정, 지지, 하중) → Solution(결과) |
| Details 창 | 선택한 항목의 속성 편집. 경계조건 범위는 뷰포트에서 면을 고른 뒤 **Apply** |
| 뷰포트 | 점·선·면·바디 선택 필터(Ctrl/Shift로 추가 선택), 표준 뷰, 단면 보기, 끌어다 놓기로 파일 가져오기 |
| 하단 창 | Messages(로그), Solution Information(수렴 그래프), Tabular Data, Graph |

### 작업 순서
1. **형상**: Geometry 탭에서 Box·Cylinder 같은 기본 형상을 만들거나 **Sketch…**로 2D를 그려 돌출·회전시킵니다.
   STEP/IGES/STL/DXF는 **Import CAD**로 가져오거나 뷰포트에 끌어다 놓으면 됩니다.
   불리언, 필렛(뷰포트에서 모서리 선택), 이동·회전·대칭·패턴도 지원합니다.
2. **재료**: Materials에서 라이브러리 재료(구조강, 알루미늄, 티타늄, 고무 등)를 추가하고 바디마다 지정합니다.
3. **메시**: Mesh 탭에서 요소 크기·차수·방법을 정합니다. 국부 크기 조절은 면을 선택한 뒤 **Sizing**을 누르면 됩니다.
4. **경계조건·하중**: 면을 선택한 상태에서 Environment 탭의 Fixed·Force·Pressure 등을 누르면 바로 적용됩니다.
5. **해석**: Solution 탭에서 해석 종류를 고르고 **Solve**(F5)를 누릅니다.
   - 정적(대변형 포함), 모달, 좌굴, 과도 동해석(음해), 양해법 동해석
   - 정상·과도 열전달, 열-구조 연성, 위상 최적화
6. **결과**: Results 탭에서 변형, 응력, 소성 변형률, 온도, 안전계수 등을 추가합니다.
   애니메이션, 프로브, 변형 배율 조절, ParaView/CSV 내보내기를 쓸 수 있습니다.

Examples 메뉴의 예제 8개(브래킷, 구멍 난 판, 모달, 좌굴, 히트싱크, 위상 최적화, 접촉, 고무)를 열고 F5만 누르면 바로 결과를 볼 수 있습니다.

## 상용 프로그램과의 호환
- CAD: STEP, IGES, BREP, STL, DXF를 가져올 수 있고 STEP, STL로 내보낼 수 있습니다.
  CATIA, NX, SolidWorks 원본 파일은 각 프로그램에서 STEP으로 저장해 가져오세요.
- 해석 파일: Abaqus `.inp`와 Nastran `.bdf`는 읽기·쓰기, ANSYS `.cdb`와 gmsh `.msh`는 읽기를 지원합니다.
  GUI에서 만든 모델을 Export → `.inp`/`.bdf`로 내보내 상용 해석기로 교차 검증할 수 있습니다.

## 다른 오픈소스와 다른 점
- 요소 계산, Newton 반복, 접촉, 양해법 루프, 위상 최적화까지 모두 GPU에서 돌아갑니다.
- 재료 모델은 응력(또는 에너지) 함수 하나만 작성하면 됩니다. 접선 강성은 자동미분으로 구합니다.
- 경계조건을 CAD 면에 지정합니다. 형상을 수정해도 기하 특징(면 중심·면적·종류)으로 같은 면을 다시 찾아 연결합니다.
- 서로 맞지 않는 메시도 dual mortar 방식으로 붙여(Bonded) 경계면 패치 테스트를 통과합니다.
- 브라우저 GUI라서 GPU 서버에서 해석을 돌리고 노트북으로 확인할 수 있습니다.
