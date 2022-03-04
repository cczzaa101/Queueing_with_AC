[locally: rebuild graph and constant files
2](#locally-rebuild-graph-and-constant-files)

[dsAlgo: 2](#dsalgo)

[Backend: 2](#backend)

[Frontend: 2](#frontend)

[On znx/bak server, deploy change: 4](#on-znxbak-server-deploy-change)

[restart mongo if it is down: 4](#restart-mongo-if-it-is-down)

[pull 4 commit changes on dsAlgo, dsJava, dsTool and plearn
4](#pull-4-commit-changes-on-dsalgo-dsjava-dstool-and-plearn)

[if it is on bak server, do server config
4](#if-it-is-on-bak-server-do-server-config)

[refresh content 4](#refresh-content)

[insert graph. 4](#insert-graph.)

[restart java 4](#restart-java)

[restart frontend: 5](#restart-frontend)

[restart wechat backend: 5](#restart-wechat-backend)

[if check_backend.py in crontab is disabled, enable it.
5](#if-check_backend.py-in-crontab-is-disabled-enable-it.)

[Testing 6](#testing)

[deployment testing: 6](#deployment-testing)

[bak testing: run simple test: 6](#bak-testing-run-simple-test)

[Frontend coding instruction 7](#frontend-coding-instruction)

[Wording change 8](#wording-change)

[1. Chinese topic name: 8](#chinese-topic-name)

[2.payment modal 9](#payment-modal)

[3. redirect page display 9](#redirect-page-display)

[4. choose topic modal 9](#choose-topic-modal)

This document has 4 parts

1 how to locally compile graph and concstant files

2 how to deploy on server

3 how to test

4 wording change guide for frontend

# locally: rebuild graph and constant files

## dsAlgo:

1.  prepare files (double check when adding new concept)

    1.  problem_contents/gsee_white_black_topic.txt This is for building
        graph

    2.  problem_contents/topic_config/\[module_name\]/syllabus/\[syllabus_type\].json
        This is for generating topic details

    3.  problem_contents/topic_config/\[module_name\]/exam_type_config.json
        This is for generating black/white concept/topic list.

    4.  external_scripts/java_support_file_utils/support_files_for_gsee/backend/
        gsee_calculus/collapsed_concept_2\_concepts.json

2.  Check content: locally run external_scripts/check_errors.py Fix
    outstanding erros

3.  locally run graph and support files

    a.  run gsee_all_update_problems_dsAlgo.sh

    b.  make sure files and images are copied correctly. Possible bug
        includes wrong folder name and wrong permission.

    c.  run gsee_calcuclus_one_script_to_update_support.sh

    d.  commit changes on dsAlgo, dsJava, dsTool and plearn

## Backend:

1.  Three problems to start with

2.  Everything else is automatic.

    a.  Make sure these files are fed correctly:

> topic_2\_concept_nodes.json
>
> spectial_topic_2\_concept_nodes.json
>
> concept_node_2\_topics.json

3.  test the first problem to see if it makes sense.

## Frontend:

1.  Follow frontend coding instructions to modify code

2.  Price/Combo price for new topic

3.  Wording change

    a.  Chinese name

    b.  动宾短语：证明 xxx，计算 xxx

    c.  为什么重要/难

    d.  内容包括

4.  Prepare diagram: <https://www.yasuotu.com/msize>

    a.  Draw from ppt

    b.  Save as jpg

    c.  Compress image, save to
        plearn/app_figs/diagram/knowledge_flowchart\_\[module\]\_\[topic\].jpg

    d.  Compress image while change width to 480\
        plearn/app_figs/diagram/knowledge_flowchart\_\[module\]\_mobile\_\[topic\].jpg

5.  prepare constant files:

topic_detail.json

concept_node_2\_topics.json

6.  make sure db is consistent automatically.

    a.  mongodb:
        db.testcalcs.update({},{\$set:{\'remainingUsage.series-limit\':1}},{multi:true})

    b.  just in case, here is the script to update user accounts.

external_scripts/update_fields_multi_topic.py

User stats and email reminder:

Search for limit-compute and mean-value and add the new topic.

# On znx/bak server, deploy change:

## restart mongo if it is down:

a.  type mongo, use plearn, show collections to see if mongo is correct

b.  if not, restart mongo

    i.  kill -9 \$(pgrep mongo)

    ii. nohup mongod \--fork \--logpath \~/log/mongod.log \--dbpath
        \~/data/db&

## pull 4 commit changes on dsAlgo, dsJava, dsTool and plearn

c.  if it is on test server, use pre\_ branches

d.  if it is on prod server, use main branches.

e.  cd dsAlgo, git pull, cd

f.  cd dsJava, git pull, cd

g.  cd dsTool, git pull, cd

h.  cd plearn, git pull, cd

## if it is on bak server, do server config

i.  cd plearm, vim common/constants.js

j.  change "is_beta_test" to true

## refresh content

k.  cd dsAlgo

l.  run ./gsee_all_update_problems_znx.sh --f (this drops collection, so
    is destructive, should do during idle time)

## insert graph. 

m.  cd dsAlgo

n.  if it is for calculus, run

> python -m
> external_scripts.gsee_calculus_one_script_for_graph_reconstruction_no_update

o.  if it is for linear_algebra, run

> python -m
> external_scripts.gsee_linear_algebra_one_script_for_graph_reconstruction_no_update

p.  if it is for prob, run

> python -m
> external_scripts.gsee_one_script_for_graph_reconstruction_no_update

## restart java

q.  sh /root/rebuild_java_backend.sh

r.  if this does work, try manual restart:

    i.  cd dsJava

    ii. cd /root/dsJava/BPonSpring && git pull && gradle build -x test
        && kill -9 \$(pgrep -f gs-rest-service-0.1.0.jar)

    iii. nohup java -Xmx40g -XX:MaxGCPauseMillis=200 -verbose:gc
         -Xloggc:/root/backend_gc_log.log -XX:+UseG1GC
         -XX:+PrintGCDetails -XX:+PrintGCDateStamps
         -XX:+UseStringDeduplication -jar
         build/libs/gs-rest-service-0.1.0.jar \> /root/backend_java.log
         &

## restart frontend: 

s.  cd plearn, ./redeploy_aliyun.sh,

t.  after redeploy is done run python3 refresh.py

## restart wechat backend: 

u.  curl \"localhost:12451/welcome\"

v.  If no response, cd plearn_wechat, ./start.sh

## if check_backend.py in crontab is disabled, enable it.

# Testing

## deployment testing:

a.  check all services is running:

    i.  wechat backend:

        1.  curl \"localhost:12451/welcome\"

        2.  forever list

> ![](doc_images/media/image1.png)

ii. mongodb:

    1.  mongo, use plearn, show collections

iii. graph, problem, java, frontend

     1.  open a browser in Incognito mode

     2.  open development tool: right click, inspect code, select
         "Network"

     3.  check "disable cache" option

     4.  use your own account,

         a.  try to login

         b.  switch to the new topic, and check:

             i.  The progress bar

             ii. The first problem

             iii. Payment page wording

## bak testing: run simple test:

b.  register a new account

    i.  username starts with your netid, + any string

    ii. password is znxznx

c.  check

    i.  first problem is correct

    ii. can finish the modified topic by doing all problems correct
        (always input "a" as answer)

    iii. redirect page wording

    iv. topic selection page wording/diagram

    v.  payment page wording

    vi. promotion page wording

# Frontend coding instruction

New topic checklist:

RedirectPageDisplay:

1.  包括所有\....

2.  高数专题： 极限，\.... (2 places)

EntrancePage.js:

1 目前，我们的内容包括：..

PromotionPage.jsx:

1 高数专题1\...

LandingQA.js:

search for all 中值定理 part

TutorSessionSummaryModal.js

renderMessage() line 1

ChooseTopicModal.js:

1.  adjust size

2.  add new topic intro

PaymentModal.js:

add new topic intro

MuiModulePage.jsx:

around line 598, add to \[{name: \'limit-compute\', cname: \'极限\'}\]

constants.js:

1.  in priceDictForServer, add new item. Example:
    \'MOREUNLIMITEDGROUP_MEAN_CALC\':2900, \'MOREUNLIMITED_MEAN_CALC\':
    4900, \'MORE3_MEAN_CALC\': 1900

2.  in itemNameDict.calc, add new item. Example:
    \'mean-value\':\[\'MORE3_MEAN_CALC\', \'MOREUNLIMITED_MEAN_CALC\',
    \'MOREUNLIMITEDGROUP_MEAN_CALC\'\] (should match last step)

3.  in moduleChineseNameDict.calc, add new item. Example:
    \'mean-value\':\[\'高数\[中值定理\]\'\],

4.  in topicTechChineseNameDict, add new item. Example:
    \'mean-value\':\'掌握中值定理相关证明\'

CouponCenter.js:

1.  search for \'\_MEAN_CALC\', add item(should match constants.js).
    Example: case \'\_SERIES_CALC\': return \'series-limit\'

2.  similarly, search for \'mean-value\', add item. Example: case
    \'mean-value\': return \'\_MEAN_CALC\'

3.  around line 214, search for \<option\> tags. add item. Example:
    \<option value = \'mean-value\' \>mean value\</option\>

AuthReducer.js:

1\. add item. Example:\
\'series-limit\':{\
pairingCode: \"\",\
pairingStatus: \"INITIAL\",\
expireTime: \"\"\
}\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

payment.controller.js

1.  in addFreeWeakness(), add item. Example:
    if(promotionType.indexOf(\'mean-value\')!==-1)return \'mean-value\'

2.  in addFreeWeakness(), add item. Example:
    validPromotions.push(\'shareBadge\'+itemSuffix +\'\_\' +
    \'series-limit\')

3.  in extractItemInfo(), add item. Example: else
    if(itemName.indexOf(\'SERIES\')!==-1) topic = \'series-limit\'

test.controller.js

1.  in switchCalculusTopic(), add to availableTopics.

2.  let usage={\....}, add new initial usage. example: \'mean-value\':1

models/test.js

1.  in remainingUsageSchema, add new topic

2.  in allowanceSchema, add new Topic

# Wording change

## 1. Chinese topic name:

![](doc_images/media/image2.png)

1.5 Initial page 动宾短语："计算积分" "证明数列极限"，"攻克积分难点"

## 2.payment modal

![](doc_images/media/image3.png)

## 3. redirect page display

![](doc_images/media/image4.png)

## 4. choose topic modal

![](doc_images/media/image5.png)
