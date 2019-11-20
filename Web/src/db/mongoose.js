const mongoose = require('mongoose')
const db = 'mongodb://localhost:27017/myapp'
mongoose.connect(db, {
        useNewUrlParser: true,
        useCreateIndex: true,
        useFindAndModify: false
    }).then(() => console.log('MongoDB Connected...'))
    .catch(err => console.log(err))