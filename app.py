from flask import Flask, request, jsonify, render_template
import pickle
import torch
from transformers import BertTokenizer, BertModel
from urllib.parse import urlparse
import re
import tld
import dns.resolver 
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# BERT 모델 로드
bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# URL 정보 추출 함수 (판단근거로 사용)
def get_url_info(url):
    url_info = {}
    
    # URL 길이
    url_info['url_len'] = len(url)
    
    #fail_silently=True로 tld unknown일 경우 0으로 나오게 일단 함
    parsed_tld = tld.get_tld(url, as_object=True, fail_silently=True, fix_protocol=True)
    # 도메인 정보 추출
    try:
        url_info['domain_len'] = len(parsed_tld.domain)
        url_info['tld'] = parsed_tld.tld
        
    except Exception as e:
        url_info['domain_len'] = 0
        url_info['tld'] = ""
        
        
    # 서브도메인 존재 여부
    def having_Sub_Domain(parsed_tld):
        if parsed_tld is not None:
            subdomain = parsed_tld.subdomain
            if subdomain == "":
                return 0
            return 1 
        return 0 
    url_info['sub_domain'] = having_Sub_Domain(parsed_tld)
    
    # 파라미터 길이
    parsed_url = urlparse(url)
    url_info['parameter_len'] = len(parsed_url.query)
    
    # IP 주소 존재 여부
    ipv4_pattern = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
    ipv6_pattern = re.compile(r'(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|'
                              r'([0-9a-fA-F]{1,4}:){1,7}:|'
                              r'([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|'
                              r'([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|'
                              r'([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|'
                              r'([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|'
                              r'([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|'
                              r'[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|'
                              r':((:[0-9a-fA-F]{1,4}){1,7}|:)|'
                              r'fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|'
                              r'::(ffff(:0{1,4}){0,1}:){0,1}'
                              r'(([0-9]{1,3}\.){3,3}[0-9]{1,3})|'
                              r'([0-9a-fA-F]{1,4}:){1,4}'
                              r':([0-9]{1,3}\.){3,3}[0-9]{1,3})')
    url_info['having_ip_address'] = 1 if ipv4_pattern.search(url) or ipv6_pattern.search(url) else 0
    
    # 프로토콜
    url_info['protocol'] = 1 if urlparse(url).scheme == "http" else 0
    
    # 비정상 URL 여부
    #hostname = parsed_url.hostname
    #url_info['abnormal_url'] = 1 if hostname and re.search(hostname, url) else 0
    # 비정상 URL 여부: DNS 조회로 도메인 유효성 검사
    hostname = parsed_url.hostname
    
    url_info['abnormal_url'] = 0
    if hostname:
        try:
            # DNS 조회를 통해 호스트 이름의 유효성을 검사
            dns.resolver.resolve(hostname, 'A')  # A 레코드를 조회하여 호스트 이름의 존재 여부 확인
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            url_info['abnormal_url'] = 1
        except Exception as e:
            url_info['abnormal_url'] = 1
    
    return url_info

# URL 구성 요소 추출 함수 정의
def parse_url_components(url):
    # URL 파싱
    parsed_url = urlparse(url)
    
    # 구성 요소 추출
    protocol = parsed_url.scheme
    domain = parsed_url.netloc
    path = parsed_url.path
    params = parsed_url.query
    subdomain = ".".join(parsed_url.netloc.split(".")[:-2])
    
    return protocol, domain, subdomain, path, params

# 각 구성 요소의 특징 추출 함수 정의
def extract_component_features(protocol, domain, subdomain, path, params):
    features = {}
    
    # 프로토콜: http/https
    features['protocol_http'] = 1 if protocol == "http" else 0
    
    # 도메인 길이
    features['domain_len'] = len(domain)
    
    # 서브도메인 존재 여부
    features['has_subdomain'] = 1 if subdomain else 0
    
    # 경로 길이
    features['path_len'] = len(path)
    
    # 파라미터 길이
    features['params_len'] = len(params)
    
    # IP 주소 포함 여부
    features['has_ip_address'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', domain) else 0
    
    return features

def standardize_url(url):
    if not url.endswith('/'):
        url = url + '/'  # 마지막에 '/' 추가
    return url

def extract_features(url):
    
    # URL 표준화
    url = standardize_url(url)
    
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    # BERT 인코딩
    inputs = tokenizer.encode_plus(url, return_tensors='pt', add_special_tokens=True, max_length=128, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        
    # 마지막 4개의 히든 레이어 평균
    token_vecs = [torch.mean(hidden_states[layer][0], dim=0) for layer in range(-4, 0)]
    bert_features = torch.stack(token_vecs).numpy().flatten()
    
    # URL 구성 요소별 특징 추출
    protocol, domain, subdomain, path, params = parse_url_components(url)
    url_component_features = extract_component_features(protocol, domain, subdomain, path, params)
    additional_features = np.array(list(url_component_features.values()))
    
    # 특징 결합
    combined_features = np.concatenate([bert_features, additional_features])
    
    return combined_features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    
    # URL 정보를 추출
    url_info = get_url_info(url)
    
    # URL을 BERT 토크나이저를 사용하여 특징 추출
    features = extract_features(url)
    
    #예측 결과
    prediction = model.predict(features.reshape(1, -1))
    
    # 로그에 prediction 값 출력
    app.logger.debug(f"Prediction value: {prediction[0]}")
    
    
    #json 형식 응답 반환
    # return jsonify({'prediction': prediction[0]})
    
    # 예측 결과와 URL 정보를 반환
    return jsonify({
        # 'prediction': prediction[0],
        'prediction': int(prediction[0]),
        'url_info': url_info
    })
    
if __name__ == '__main__':
    app.run(debug=True)