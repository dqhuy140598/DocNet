const mongoose = require('mongoose');

const documentSchema = mongoose.Schema({
    documentTitle: {
        type:String,
        require:true
    },
    content:{
        type:String,
        require:true,
    },
    imageFile:{
        type:String,
    },
    keyPhrase:{
        type:[String],
    }
})

const document = mongoose.model('Document',documentSchema)

module.exports = document