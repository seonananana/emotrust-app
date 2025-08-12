// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";

contract EmpathyNFT extends ERC721URIStorage {
    uint256 private _nextTokenId;

    constructor() ERC721("EmpathyNFT", "EMO") {}

    /// 팀 개발용: 테스트넷/로컬에서 누구나 민팅 가능 (필요시 onlyOwner로 바꾸면 됨)
    function safeMint(address to, string memory tokenURI_) public returns (uint256) {
        uint256 tokenId = ++_nextTokenId;
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, tokenURI_);
        return tokenId;
    }
}
