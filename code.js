const {exec} = require("child_process");
const {setInterval} = require("timers/promises");
const fs = require('fs')
let N = 90
let ACTIVE = false
let STATE = "TRAIN"

const PY_TRAIN_FILE = "train.py"
const PY_VALIDATION_FILE = "validate.py"
const VALIDATION_CSV = "validation.csv"
const MODEL_FILE = "auto.h5"
const CHECK_INTERVAL = 1000;
const N_LIMIT = 128;

function runCommand(){
    let cp = exec(`py ${PY_TRAIN_FILE} ${N}`)
    ACTIVE = true;
    cp.stdout.on("data",console.log)
    cp.on("close",(code, signal)=>{
        N+=1
        ACTIVE = false
        STATE = "VALIDATE"
    })
    return cp
}

async function checkIfRunning(){
    for await(const startTime of setInterval(CHECK_INTERVAL, Date.now())) {
        if(!ACTIVE){
            switch(STATE){
                case "TRAIN":
                    console.log(`>>> RUNNING ${N} <<<`)
                    runCommand()
                    break;
                case "VALIDATE": 
                    if(N != 0){
                        console.log(`>>> VALIDATE ${N-1} <<<`)
                        checkModel()
                    }
                    break;
            }
        }
        if(N == N_LIMIT)
            break;
    }
}
function checkModel(){
    let cp = exec(`py ${PY_VALIDATION_FILE}`)
    ACTIVE = true
    cp.stdout.on("data",(chunk)=>{
        console.log(chunk)
        fs.appendFile(VALIDATION_CSV,`models/model_${N-1}.h5,${chunk}`,()=>{})
    })
    cp.on("close",(code, signal)=>{
        if(fs.existsSync(MODEL_FILE)){
            fs.copyFile(MODEL_FILE,`models/model_${N-1}.h5`,()=>{})
        }else{
            console.warn(`${MODEL_FILE} not found.`)
        }
        ACTIVE = false
        STATE = "TRAIN"
    })
}


// RUN
checkIfRunning().then(e=>{
    console.log("FINISH")
})