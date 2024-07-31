from sqlglot import parse_one, exp
import sqlglot.expressions as exp
import sqlglot
import json
import os
import re
import copy
def remove_spaces_between_chars(string, char1, char2):
  # create a pattern that matches char1 followed by any number of spaces followed by char2
  pattern = char1 + "\s+" + char2
  # replace the spaces with an empty string
  result = re.sub(pattern, char1 + char2, string)
  return result
class statement_embdi():
    state_embdi_dir={}
    index=0
    aggregate_function=["AVG","COUNT","MAX","MIN","SUM","COUNT_BIG","GROUPING","BINARY_CHECKSUM","CHECKSUM_AGG","CHECKSUM","STDEV","STDEVP","VAR","VARP","FIRST","LAST","DATEDIFF","JULIANDAY","CAST","WEEK","YEAR","MONTH","DATE_FORMAT","EXTRACT","TIMESTAMPDIFF","NOW","LENGTH","GROUP_CONCAT","ROW_NUMBER","COALESCE","SUBSTR","REVERSE","CONCAT","ROUND","GETDATE","CURRENT_DATE","FILTER","CURRENT_TIMESTAMP","OVER","ABS","HOUR","TIMESTAMP","LOWER","STDDEV"]
    condition=[">","<","=","IN","EXISTS","BETWEEN","NOT",">=","<=","<>"]
    textformat=["INT","STRING","BOOLEAN","BIT","TINYINT","SMALLINT","FLOAT","DOUBLE","CHAR","DATE","DATETIME","TIME","IDENTIFIER","NUMBER"]
    arithmetic_operation=["PLUS","STAR","DASH","SLASH","MOD"]
    def __init__(self):
        self.tokenizer=sqlglot.tokens.Tokenizer()
        self.data11=[]
        self.data22=[]
        if os.path.exists("state_embdi.json"):
            with open("state_embdi.json", "r") as f:
                d=json.load(f)
                statement_embdi.index=d[0]
                statement_embdi.state_embdi_dir = d[1]
    def find_embdi(self,state):
        if state in statement_embdi.state_embdi_dir.keys():
            return statement_embdi.state_embdi_dir[state]
        else:
            statement_embdi.state_embdi_dir[state]=statement_embdi.index
            statement_embdi.index+=1
            return statement_embdi.index-1
    def pre_process(self,sql):
        sql=remove_spaces_between_chars(sql,"<","=")
        sql=remove_spaces_between_chars(sql,"<",">")
        sql=remove_spaces_between_chars(sql,">","=")
        sql=remove_spaces_between_chars(sql,"=",">")
        sql=sql.strip()
        #sql=sql.replace("\"","\'")
        return sqlglot.transpile(sql, write="sqlite", pretty=False)[0]
    def write_down(self):
        with open("state_embdi.json", "w") as f:
            d=[statement_embdi.index,statement_embdi.state_embdi_dir]
            json.dump(d,f)
    def token_embdi(self,sqll):
        data=[]
        sqll=self.pre_process(sqll)
        expression=parse_one(sqll)
        for token in expression.walk(bfs=False):
            if isinstance(token[1],exp.Select): location="select"
            if isinstance(token[1],exp.From): location="from"
            if isinstance(token[1],exp.Where): location="where"
            if isinstance(token[1],exp.Join): location="join"
            if isinstance(token[0],exp.Column):
                if hasattr(token[0],"table")and token[0].table: 
                    data.append((str(token[0].table),location+"TABLE"))
                data.append((str(token[0].this),location+"COLUMN"))
                continue
            if isinstance(token[0],exp.Table):
                data.append((str(token[0].this),location+"TABLE"))
            if hasattr(token[1],"alias") and str(token[1].alias)==str(token[0]):
                data.append((str(token[0].this),location+"ALIASNAME"))
        tokens=self.tokenizer.tokenize(sqll)
        data2=[(token.text,str(token.token_type)[10:]) for token in tokens]
        # self.data11=copy.deepcopy(data)
        # self.data22=copy.deepcopy(data2)
        # print(data)
        # print(data2)
        count=0
        temp=0
        for j in data2:
            if j[1]=="VAR" and not j[0] in self.aggregate_function:temp+=1
        # print(temp,len(data))
        state_embdi=[]
        flag=False
        for index in range(len(data2)):
            i=data2[index]
            # if i[1]=="WITHIN_GROUP": 
            #     print(sqll)
            #     print(data)
            #     print(data2)
            #     flag=True
            if i[0] in self.aggregate_function:
                #data2[data2.index(i)]=(i[0],"AGGFUN")
                state_embdi.append((i[0],"AFFFUN",self.find_embdi("aggfun".upper())))
            elif i[0].upper() in self.condition:
                #data2[data2.index(i)]= (i[0],"CONDITION")
                state_embdi.append((i[0],"CONDITION",self.find_embdi("condition".upper())))
            elif temp!=len(data) and count<len(data) and i[0] == data[count][0] and i[1]!="VAR":
                state_embdi.append((i[0],data[count][1],self.find_embdi(data[count][1].upper())))
                count+=1
            elif i[1]=="VAR":
                #data2[data2.index(i)]=data[count]
                state_embdi.append((i[0],data[count][1],self.find_embdi(data[count][1].upper())))
                count+=1
            elif i[1].upper() in self.textformat:
                state_embdi.append((i[0],"LITERAL",self.find_embdi("literal".upper())))
            elif i[1] in self.arithmetic_operation:
                state_embdi.append((i[0],"ARITH",self.find_embdi("arith".upper())))
            else:
                state_embdi.append((i[0],i[1],self.find_embdi(i[1].upper())))
        for index in range(len(data2)):
            i=data2[index]
            if i[0]=="*" :
                if (index>=1 and (state_embdi[index-1][1]=="LITERAL" or "COLUMN" in state_embdi[index-1][1] or "TABLE" in state_embdi[index-1][1]) or (state_embdi[index+1][1]=="LITERAL" or "COLUMN" in state_embdi[index+1][1] or "TABLE" in state_embdi[index+1][1])):
                    continue
                else:
                    state_embdi[index]=(i[0],i[1],self.find_embdi(i[1].upper()))
        # if(flag):print(state_embdi)
        return state_embdi