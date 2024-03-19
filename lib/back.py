# import xlrd  #可替換為pandas
import os
import pandas as pd
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import logging

pos_EtoC = {
        "A":"非謂形容詞",
        "Caa":"對等連接詞",
        "Cab":"連接詞(等等)",
        "Cba":"連接詞(的話)",
        "Cbb":"關聯連接詞",
        "D":"副詞",
        "Da":"數量副詞",
        "Dfa":"動詞前程度副詞",
        "Dfb":"動詞後程度副詞",
        "Di":"時態標記",
        "Dk":"句副詞",
        "DM":"定量式",
        "I":"感嘆詞",
        "Na":"普通名詞",
        "Nb":"專有名詞",
        "Nc":"地方詞",
        "Ncd":"位置詞",
        "Nd":"時間詞",
        "Nep":"指代定詞",
        "Neqa":"數量定詞",
        "Neqb":"後置數量定詞",
        "Nes":"特指定詞",
        "Neu":"數詞定詞",
        "Nf":"量詞",
        "Ng":"後置詞",
        "Nh":"代名詞",
        "Nv":"名物化動詞",
        "P":"介詞",
        "T":"語助詞",
        "VA":"動作不及物動詞",
        "VAC":"動作使動動詞",
        "VB":"動作類及物動詞",
        "VC":"動作及物動詞",
        "VCL":"動作接地方賓語動詞",
        "VD":"雙賓動詞",
        "VF":"動作謂賓動詞",
        "VE":"動作句賓動詞",
        "VG":"分類動詞",
        "VH":"狀態不及物動詞",
        "VHC":"狀態使動動詞",
        "VI":"狀態類及物動詞",
        "VJ":"狀態及物動詞",
        "VK":"狀態句賓動詞",
        "VL":"狀態謂賓動詞",
        "V_2":"有",
        "DE":"的之得地",
        "SHI":"是",
        "FW":"外文",
        "COLONCATEGORY":"冒號",
        "COMMACATEGORY":"逗號",
        "DASHCATEGORY":"破折號",
        "DOTCATEGORY":"點號",
        "ETCCATEGORY":"刪節號",
        "EXCLAMATIONCATEGORY":"驚嘆號",
        "PARENTHESISCATEGORY":"括號",
        "PAUSECATEGORY":"頓號",
        "PERIODCATEGORY":"句號",
        "QUESTIONCATEGORY":"問號",
        "SEMICOLONCATEGORY":"分號",
        "SPCHANGECATEGORY":"雙直線",
        "WHITESPACE":"空白"}

# Initialize drivers
ws_driver  = CkipWordSegmenter(model="bert-base")
pos_driver = CkipPosTagger(model="bert-base")
ner_driver = CkipNerChunker(model="bert-base")

# Use GPU:0
ws_driver = CkipWordSegmenter(device=0)


from flask import Flask, request
# app = Flask(__name__,template_folder="templates")
app = Flask(__name__)
# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():

    file_data = request.files['file']

    logging.debug('File received:', file_data.filename)
    # filename = file_data.filename
    # file.save(os.path.join(FILE_DIR,filename))
    if not file_data:
        logging.error('No file data received')
        return '未上傳檔案，或檔案格式不符'
    
        
    # 這行將文件轉為流，在xlrd中打開
    # f = file_data.read()
    # excel_file = xlrd.open_workbook(file_contents=f)

    df = pd.read_excel(file_data, usecols=["Content"])
    text = []
    for i in range(len(df)):
        text.append(df["Content"].iloc[i])

    ws  = ws_driver(text, use_delim=True)
    ner = ner_driver(text, use_delim=True)
    pos = pos_driver(ws, delim_set='\n\t')
    

    def pack_ws_pos_sentece(sentence_ws, sentence_pos):
        assert len(sentence_ws) == len(sentence_pos)
        res = []
        for word_ws, word_pos in zip(sentence_ws, sentence_pos):
            res.append(f"{word_ws}({word_pos})")
        return "\u3000".join(res)

    pos_ch = []
    out_sentence_pos = []
    for sentence, sentence_ws, sentence_pos, sentence_ner in zip(text, ws, pos, ner):
        # print(sentence)
        out_sentence_pos.append(sentence)
        for i in range(len(sentence_pos)):
            pos_ch.append(pos_EtoC[sentence_pos[i]])
        # print(pack_ws_pos_sentece(sentence_ws, pos_ch))
        # print()
        pos = pack_ws_pos_sentece(sentence_ws, pos_ch)
        out_sentence_pos.append(pos)
        out_sentence_pos.append('\n')
        pos_ch.clear()
    
    return '\n'.join(out_sentence_pos)

if __name__ == '__main__':
    app.run(debug=True)
